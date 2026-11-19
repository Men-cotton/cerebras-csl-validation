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
u16 setup_clock[n=6];

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
    setup_clock[0] = tsc[0];
    setup_clock[1] = tsc[1];
    setup_clock[2] = tsc[2];
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
    setup_clock[3] = tsc[0];
    setup_clock[4] = tsc[1];
    setup_clock[5] = tsc[2];
    frame in [0,12) {
        u32 trial = (u32)rx[];
        u32 tag = (u32)rx[];
        i in [0,64) panel[i] <- rx[];
        clock[0] = tsc[0];
        clock[1] = tsc[1];
        clock[2] = tsc[2];
        i in [0,ROWS*4) result[i] <- 0.0;
        block in [0,ROWS/64) {
            row_work(block*64+0);
            row_work(block*64+1);
            row_work(block*64+2);
            row_work(block*64+3);
            row_work(block*64+4);
            row_work(block*64+5);
            row_work(block*64+6);
            row_work(block*64+7);
            row_work(block*64+8);
            row_work(block*64+9);
            row_work(block*64+10);
            row_work(block*64+11);
            row_work(block*64+12);
            row_work(block*64+13);
            row_work(block*64+14);
            row_work(block*64+15);
            row_work(block*64+16);
            row_work(block*64+17);
            row_work(block*64+18);
            row_work(block*64+19);
            row_work(block*64+20);
            row_work(block*64+21);
            row_work(block*64+22);
            row_work(block*64+23);
            row_work(block*64+24);
            row_work(block*64+25);
            row_work(block*64+26);
            row_work(block*64+27);
            row_work(block*64+28);
            row_work(block*64+29);
            row_work(block*64+30);
            row_work(block*64+31);
            row_work(block*64+32);
            row_work(block*64+33);
            row_work(block*64+34);
            row_work(block*64+35);
            row_work(block*64+36);
            row_work(block*64+37);
            row_work(block*64+38);
            row_work(block*64+39);
            row_work(block*64+40);
            row_work(block*64+41);
            row_work(block*64+42);
            row_work(block*64+43);
            row_work(block*64+44);
            row_work(block*64+45);
            row_work(block*64+46);
            row_work(block*64+47);
            row_work(block*64+48);
            row_work(block*64+49);
            row_work(block*64+50);
            row_work(block*64+51);
            row_work(block*64+52);
            row_work(block*64+53);
            row_work(block*64+54);
            row_work(block*64+55);
            row_work(block*64+56);
            row_work(block*64+57);
            row_work(block*64+58);
            row_work(block*64+59);
            row_work(block*64+60);
            row_work(block*64+61);
            row_work(block*64+62);
            row_work(block*64+63);
        }
        clock[3] = tsc[0];
        clock[4] = tsc[1];
        clock[5] = tsc[2];
        if (tag != 0x41474752:u32) trial = 0:u32;
        tx[] <- trial;
        tx[] <- (u32)X;
        i in [0,6) tx[] <- (u32)clock[i];
        i in [0,6) tx[] <- (u32)setup_clock[i];
        i in [0,ROWS*4) tx[] <- result[i];
    }
}
