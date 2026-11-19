// A contiguous row partition. Stage the CSR graph before the first frame.
import arch_hdr;
xp param X;
xp param ROWS;
socket rx [[iq=0]];
socket tx;
xp rowptr[n=ROWS+1];
xp sources[n=85];
sp weights[n=85];
sp panel[n=1869];
sp result[n=ROWS*7];
u16 tsc[3] address(CE_TSC_ADDR);
u16 ctrl[1] address(CE_PERF_CNTR_CTRL_ADDR);
u16 clock[n=6];
u16 setup[n=6];

inline function accumulate(xp dst, xp e) {
    xp src = sources[e];
    sp scale = weights[e];
    f in [0,7) result[dst+f] <+- panel[src+f]*scale;
}

inline function aggregate_row(xp r) {
            xp dst = r*7;
            xp e = rowptr[r];
            xp end = rowptr[r+1];
            if (end == e+4) {
                accumulate(dst,e);
                accumulate(dst,e+1);
                accumulate(dst,e+2);
                accumulate(dst,e+3);
            } else {
                while (e < end) {
                    accumulate(dst,e);
                    e += 1;
                }
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
    e in [0,edges) sources[e] = sources[e]*7;
    ctrl[0] = ctrl[0] | 4;
    setup[3] = tsc[0]; setup[4] = tsc[1]; setup[5] = tsc[2];
    frame in [0,129) {
        u32 trial = (u32)rx[];
        u32 tag = (u32)rx[];
        i in [0,1869) panel[i] <- rx[];
        clock[0] = tsc[0];
        clock[1] = tsc[1];
        clock[2] = tsc[2];
        i in [0,ROWS*7) result[i] <- 0.0;
        block in [0,ROWS/16) {
            aggregate_row(block*16+0);
            aggregate_row(block*16+1);
            aggregate_row(block*16+2);
            aggregate_row(block*16+3);
            aggregate_row(block*16+4);
            aggregate_row(block*16+5);
            aggregate_row(block*16+6);
            aggregate_row(block*16+7);
            aggregate_row(block*16+8);
            aggregate_row(block*16+9);
            aggregate_row(block*16+10);
            aggregate_row(block*16+11);
            aggregate_row(block*16+12);
            aggregate_row(block*16+13);
            aggregate_row(block*16+14);
            aggregate_row(block*16+15);
        }
        clock[3] = tsc[0];
        clock[4] = tsc[1];
        clock[5] = tsc[2];
        if (tag != 0x41474752:u32) trial = 0:u32;
        tx[] <- trial;
        tx[] <- (u32)X;
        i in [0,6) tx[] <- (u32)clock[i];
        i in [0,6) tx[] <- (u32)setup[i];
        i in [0,ROWS*7) tx[] <- result[i];
    }
}
