import arch_hdr;
xp param X,P,R,F,Q,E,RELU,FRAMES;
socket rx [[iq=0]];
socket tx;
sp weight[n=(2*F+1)*Q];
sp roots[n=R*F];
sp sum[n=R*F];
sp table[n=E*F];
sp out[n=R*Q];
sp inverse[n=R];
u32 ids[n=R];
u32 selectors[n=40];
u32 ignored[n=40];
u16 clocks[n=12];
u16 timer[n=6];
u16 tsc[3] address(CE_TSC_ADDR);
u16 control[1] address(CE_PERF_CNTR_CTRL_ADDR);
inline function stamp(xp i) {
    clocks[i]=tsc[0]; clocks[i+1]=tsc[1]; clocks[i+2]=tsc[2];
}

inline function readtimer(xp dst) {
    xp stable=0;
    while (stable==0) {
        u16 high=tsc[2]; u16 mid=tsc[1]; u16 low=tsc[0];
        u16 mid2=tsc[1]; u16 high2=tsc[2];
        if (high==high2 && mid==mid2) {
            timer[dst]=low; timer[dst+1]=mid; timer[dst+2]=high;
            stable=1;
        }
    }
}
inline function calibrate() {
    sample in [0,5) {
        u32 ticket=(u32)rx[]; u32 check=(u32)rx[];
        if (X==0) {
            readtimer(0);
            tx[] <- ticket; tx[] <- check;
        }
        u32 end=(u32)rx[]; u32 confirm=(u32)rx[];
        if (X==0) {
            readtimer(3);
            tx[] <- end; tx[] <- ticket; k in [0,6) tx[] <- (u32)timer[k];
        }
    }
}

main {
    control[0]=control[0]|4;
    calibrate();
    frame in [0,FRAMES) {
        u32 gate=(u32)rx[];
        u32 check=(u32)rx[];
        if (X==0) {
            readtimer(0);
            tx[] <- gate; tx[] <- check;
        }
        xp padding=(xp)(u32)rx[];
        if (padding!=0) { u32 pad=(u32)rx[]; }
        u32 tag=(u32)rx[];
        if (frame==0) { k in [0,(2*F+1)*Q) weight[k] <- rx[]; }
        stamp(0);
        xp rows=0;
        who in [0,P) {
            xp count=(xp)(u32)rx[];
            if (who==X) {
                rows=count;
                k in [0,count) ids[k] <- (u32)rx[];
                k in [0,count) inverse[k] <- rx[];
                k in [0,count*F) roots[k] <- rx[];
            } else {
                k in [0,count) table[k] <- rx[];
                k in [0,count) table[k] <- rx[];
                k in [0,count*F) table[k] <- rx[];
            }
        }
        k in [0,R*F) sum[k] <- 0.0;
        stamp(3);
        xp steps=(xp)(u32)rx[];
        u32 edges=0;
        block in [0,steps) {
            xp entries=(xp)(u32)rx[];
            k in [0,entries*F) table[k] <- rx[];
            xp mine=0;
            who in [0,P) {
                xp count=(xp)(u32)rx[];
                if (who==X) {
                    mine=count;
                    edges+=(u32)count;
                    k in [0,count) selectors[k] <- (u32)rx[];
                } else {
                    k in [0,count) ignored[k] <- (u32)rx[];
                }
            }
            k in [0,mine) {
                u32 ref=selectors[k];
                xp dst=(xp)(ref>>16)*F;
                xp src=(xp)(ref&0xffff)*F;
                ch in [0,F) sum[dst+ch] <- sum[dst+ch]+table[src+ch];
            }
        }
        stamp(6);
        row in [0,rows) {
            xp to=row*Q;
            xp from=row*F;
            o in [0,Q) out[to+o] <- weight[2*F*Q+o];
            ch in [0,F) {
                sp x=roots[from+ch];
                sp m=sum[from+ch]*inverse[row];
                if (x!=0.0) { o in [0,Q) out[to+o] <+- weight[ch*Q+o]*x; }
                if (m!=0.0) { o in [0,Q) out[to+o] <+- weight[F*Q+ch*Q+o]*m; }
            }
            if (RELU!=0) { o in [0,Q) { if (out[to+o]<0.0) out[to+o]=0.0; } }
        }
        stamp(9);
        tx[] <- tag; tx[] <- (u32)X; tx[] <- (u32)rows; tx[] <- edges;
        k in [0,12) tx[] <- (u32)clocks[k];
        k in [0,R) tx[] <- ids[k];
        k in [0,R*Q) tx[] <- out[k];
        u32 finish=(u32)rx[];
        u32 confirm=(u32)rx[];
        if (X==0) {
            readtimer(3);
            tx[] <- finish; tx[] <- gate;
            k in [0,6) tx[] <- (u32)timer[k];
        }
    }
    calibrate();
}
