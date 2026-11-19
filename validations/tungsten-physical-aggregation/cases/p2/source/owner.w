// A contiguous row partition. Stage the CSR graph before the first frame.
import arch_hdr;
xp param X;
xp param ROWS;
socket rx [[iq=0]];
socket tx;
xp rowptr[n=ROWS+1];
xp sources[n=ROWS*4];
sp weights[n=ROWS*4];
sp panel[n=64];
sp result[n=ROWS*4];
u16 tsc[3] address(CE_TSC_ADDR);
u16 ctrl[1] address(CE_PERF_CNTR_CTRL_ADDR);
u16 clock[n=6];

main {
    xp edges = 0;
    r in [0,ROWS) {
        rowptr[r] = edges;
        xp global_r = X*ROWS+r;
        if ((global_r & 7) != 0) {
            xp src = global_r*3;
            sources[edges] = src & 15;
            sources[edges+1] = (src+1) & 15;
            sources[edges+2] = (src+1) & 15;
            sources[edges+3] = (src+7) & 15;
            weights[edges] = 0.5;
            weights[edges+1] = -0.25;
            weights[edges+2] = 0.75;
            weights[edges+3] = -1.0;
            edges += 4;
        }
    }
    rowptr[ROWS] = edges;
    ctrl[0] = ctrl[0] | 4;
    frame in [0,3) {
        u32 trial = (u32)rx[];
        u32 tag = (u32)rx[];
        i in [0,64) panel[i] <- rx[];
        clock[0] = tsc[0];
        clock[1] = tsc[1];
        clock[2] = tsc[2];
        i in [0,ROWS*4) result[i] <- 0.0;
        r in [0,ROWS) {
            xp dst = r*4;
            xp e = rowptr[r];
            while (e < rowptr[r+1]) {
                xp src = sources[e]*4;
                sp scale = weights[e];
                f in [0,4) result[dst+f] <+- panel[src+f]*scale;
                e += 1;
            }
        }
        clock[3] = tsc[0];
        clock[4] = tsc[1];
        clock[5] = tsc[2];
        if (tag != 0x41474752:u32) trial = 0:u32;
        tx[] <- trial;
        tx[] <- (u32)X;
        i in [0,6) tx[] <- (u32)clock[i];
        i in [0,ROWS*4) tx[] <- result[i];
    }
}
