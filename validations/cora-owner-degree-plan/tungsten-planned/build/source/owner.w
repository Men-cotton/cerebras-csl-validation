// A contiguous row partition. Stage the CSR graph before the first frame.
import arch_hdr;
xp param X;
xp param ROWS;
socket rx [[iq=0]];
socket tx;
xp rowptr[n=ROWS+1];
xp sources[n=85];
sp weights[n=85];
sp panel[n=1068];
sp result[n=ROWS*4];
u16 tsc[3] address(CE_TSC_ADDR);
u16 ctrl[1] address(CE_PERF_CNTR_CTRL_ADDR);
u16 clock[n=6];
u16 setup[n=6];

inline function accumulate(xp dst, xp e) {
    xp src = sources[e];
    sp scale = weights[e];
    f in [0,4) result[dst+f] <+- panel[src+f]*scale;
}

xp degree[n=ROWS];
xp planned_sources[n=ROWS*4];
sp planned_weights[n=ROWS*4];

inline function planned(xp dst, xp i) {
    xp src = planned_sources[i];
    sp scale = planned_weights[i];
    f in [0,4) result[dst+f] <+- panel[src+f]*scale;
}

inline function row_work(xp r) {
    xp dst = r*4;
    if (degree[r] == 4) {
        planned(dst,dst);
        planned(dst,dst+1);
        planned(dst,dst+2);
        planned(dst,dst+3);
    } else if (degree[r] > 0) {
        xp e = rowptr[r];
        xp end = rowptr[r+1];
        while (e < end) { accumulate(dst,e); e += 1; }
    }
}

main {
    owner in [0,4) {
        i in [0,17) { u32 v = (u32)rx[]; if (owner == X) rowptr[i] = (xp)v; }
        i in [0,85) { u32 v = (u32)rx[]; if (owner == X) sources[i] = (xp)v; }
        i in [0,85) { sp v = rx[]; if (owner == X) weights[i] = v; }
    }
    xp edges = rowptr[ROWS];
    ctrl[0] = ctrl[0] | 4;
    setup[0] = tsc[0]; setup[1] = tsc[1]; setup[2] = tsc[2];
    e in [0,edges) sources[e] = sources[e]*4;
    r in [0,ROWS) {
        xp first = rowptr[r];
        degree[r] = rowptr[r+1]-first;
        if (degree[r] == 4) {
            j in [0,4) {
                planned_sources[r*4+j] = sources[first+j];
                planned_weights[r*4+j] = weights[first+j];
            }
        }
    }
    ctrl[0] = ctrl[0] | 4;
    setup[3] = tsc[0]; setup[4] = tsc[1]; setup[5] = tsc[2];
    frame in [0,12) {
        u32 trial = (u32)rx[];
        u32 tag = (u32)rx[];
        i in [0,1068) panel[i] <- rx[];
        clock[0] = tsc[0];
        clock[1] = tsc[1];
        clock[2] = tsc[2];
        i in [0,ROWS*4) result[i] <- 0.0;
        block in [0,ROWS/16) {
            row_work(block*16+0);
            row_work(block*16+1);
            row_work(block*16+2);
            row_work(block*16+3);
            row_work(block*16+4);
            row_work(block*16+5);
            row_work(block*16+6);
            row_work(block*16+7);
            row_work(block*16+8);
            row_work(block*16+9);
            row_work(block*16+10);
            row_work(block*16+11);
            row_work(block*16+12);
            row_work(block*16+13);
            row_work(block*16+14);
            row_work(block*16+15);
        }
        clock[3] = tsc[0];
        clock[4] = tsc[1];
        clock[5] = tsc[2];
        if (tag != 0x41474752:u32) trial = 0:u32;
        tx[] <- trial;
        tx[] <- (u32)X;
        i in [0,6) tx[] <- (u32)clock[i];
        i in [0,6) tx[] <- (u32)setup[i];
        i in [0,ROWS*4) tx[] <- result[i];
    }
}
