/* Ground-truth dumper: calls the REAL Snack DSP routines (from libsnack.dylib)
 * and writes out each intermediate stage of the formant pipeline so we can diff
 * them against the Python port stage by stage.
 *
 * Replicates the Sound* wrappers Fdownsample()/highpass()/lpc_poles() inline,
 * but every actual computation (dwnsamp, do_fir, lc_lin_fir, lpcbsa, formant)
 * is the genuine Snack object code.
 *
 * Usage: snack_dump <input_i16.raw> <samprate> <lpc_ord> <outdir>
 *   input_i16.raw : raw little-endian int16 mono samples (post read_wav + int16 quantize)
 * Writes in <outdir>: ds.raw (downsampled i16), hp.raw (highpassed i16),
 *   poles.txt (one line per frame: rms nform f0 b0 f1 b1 ...)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAXORDER 30

/* Snack prototypes (resolved from libsnack.dylib) */
extern int  ratprx(double a, int *k, int *l, int qlim);
extern int  lc_lin_fir(double fc, int *nf, double coef[]);
extern int  get_abs_maximum(short *d, int n);
extern int  dwnsamp(short *buf, int in_samps, short **buf2, int *out_samps,
                    int insert, int decimate, int ncoef, short ic[],
                    int *smin, int *smax);
extern void do_fir(short *buf, int in_samps, short *bufo, int ncoef,
                   short ic[], int invert);
extern int  lpcbsa(int np, double lpc_stabl, int wind, short *data, double *lpc,
                   double *rho, double *nul1, double *nul2, double *energy,
                   double preemp);
extern int  formant(int lpc_order, double s_freq, double *lpca, int *n_form,
                    double *freq, double *band, int init);

static short *read_i16(const char *path, int *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open input"); exit(1); }
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    int cnt = bytes / 2;
    short *buf = malloc(sizeof(short) * cnt);
    fread(buf, sizeof(short), cnt, f);
    fclose(f);
    *n = cnt;
    return buf;
}

static void write_i16(const char *path, short *buf, int n) {
    FILE *f = fopen(path, "wb");
    fwrite(buf, sizeof(short), n, f);
    fclose(f);
}

static double integerize(double t, double freq) {
    int i = (int)(0.5 + freq * t);
    return ((double)i) / freq;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s in.raw samprate lpc_ord outdir\n", argv[0]); return 1; }
    fprintf(stderr, "[stage] main entered\n"); fflush(stderr);
    int n_in;
    short *sig = read_i16(argv[1], &n_in);
    fprintf(stderr, "[stage] read %d samples\n", n_in); fflush(stderr);
    double samprate = atof(argv[2]);
    int lpc_ord = atoi(argv[3]);
    const char *outdir = argv[4];
    char path[1024];

    /* ---- Fdownsample (ds_freq = 10000) ---- */
    double ds_freq = 10000.0;
    short *cur = sig;
    int n_cur = n_in;
    double cur_rate = samprate;

    if (ds_freq < samprate) {
        int insert, decimate;
        double ratio = ds_freq / samprate;
        ratprx(ratio, &insert, &decimate, 10);
        fprintf(stderr, "[stage] ratprx: insert=%d decimate=%d\n", insert, decimate); fflush(stderr);
        double ratio_t = (double)insert / (double)decimate;
        if (ratio_t <= 0.99) {
            double freq2 = ratio_t * samprate;
            double beta = (0.5 * freq2) / (insert * samprate);
            int ncoeff = 127; double b[256]; short ic[256];
            lc_lin_fir(beta, &ncoeff, b);
            double maxi = (1 << 15) - 1;
            int j = (ncoeff / 2) + 1, ncoefft = 0;
            for (int i = 0; i < j; i++) { ic[i] = (int)(0.5 + maxi * b[i]); if (ic[i]) ncoefft = i + 1; }
            fprintf(stderr, "[stage] lc_lin_fir done ncoefft=%d, calling dwnsamp\n", ncoefft); fflush(stderr);
            short *bufout = NULL; int out_samps, smin, smax;
            dwnsamp(cur, n_cur, &bufout, &out_samps, insert, decimate, ncoefft, ic, &smin, &smax);
            cur = bufout; n_cur = out_samps; cur_rate = freq2;
        }
    }
    fprintf(stderr, "[stage] downsample done: n_cur=%d rate=%.1f\n", n_cur, cur_rate); fflush(stderr);
    snprintf(path, sizeof path, "%s/ds.raw", outdir);
    write_i16(path, cur, n_cur);

    /* ---- highpass ---- */
    int LCSIZ = 101, len = 1 + (LCSIZ / 2);
    short *lcf = malloc(sizeof(short) * len);
    double fn = M_PI * 2.0 / (LCSIZ - 1);
    double scale = 32767.0 / (0.5 * LCSIZ);
    for (int i = 0; i < len; i++) lcf[i] = (short)(scale * (0.5 + 0.4 * cos(fn * (double)i)));
    short *hp = malloc(sizeof(short) * n_cur);
    do_fir(cur, n_cur, hp, len, lcf, 1);
    fprintf(stderr, "[stage] highpass done\n"); fflush(stderr);
    snprintf(path, sizeof path, "%s/hp.raw", outdir);
    write_i16(path, hp, n_cur);

    /* ---- lpc_poles (lpc_type == 1: stabilized covariance) ---- */
    double wdur = 0.025;
    double preemp = exp(-62.831853 * 90.0 / cur_rate);
    wdur = integerize(wdur, cur_rate);
    double frame_int = integerize(0.001, cur_rate);   /* 1 ms frame shift */
    int nfrm = 1 + (int)((((double)n_cur / cur_rate) - wdur) / frame_int);
    int size = (int)(0.5 + wdur * cur_rate);
    int step = (int)(0.5 + frame_int * cur_rate);

    snprintf(path, sizeof path, "%s/poles.txt", outdir);
    FILE *pf = fopen(path, "w");
    fprintf(stderr, "[stage] lpc_poles start: nfrm=%d size=%d step=%d preemp=%.6f\n", nfrm, size, step, preemp); fflush(stderr);
    int init = 1;
    double lpca[MAXORDER + 1], frp[MAXORDER], bap[MAXORDER], energy;
    for (int fr = 0; fr < nfrm; fr++) {
        if (fr % 200 == 0) { fprintf(stderr, "  frame %d\n", fr); fflush(stderr); }
        short *datap = hp + fr * step;
        int nform = 0;
        lpcbsa(lpc_ord, 70.0, size, datap, lpca, NULL, NULL, NULL, &energy, preemp);
        if (energy > 1.0) {
            formant(lpc_ord, cur_rate, lpca, &nform, frp, bap, init);
            init = 0;
        } else {
            nform = 0; init = 1;
        }
        fprintf(pf, "%.6f %d", energy, nform);
        for (int i = 0; i < nform; i++) fprintf(pf, " %.4f %.4f", frp[i], bap[i]);
        fprintf(pf, "\n");
    }
    fclose(pf);
    fprintf(stderr, "rate=%.1f n_ds=%d nfrm=%d size=%d step=%d\n", cur_rate, n_cur, nfrm, size, step);
    return 0;
}
