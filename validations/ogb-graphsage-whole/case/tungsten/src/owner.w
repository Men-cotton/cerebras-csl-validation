import arch_hdr;
xp param X;
xp param P;
xp param R;
xp param F;
xp param Q;
xp param E;
xp param K;
xp param RELU;
socket rx [[iq=0]];
socket tx;
sp matrix[n=(2*F+1)*Q];
sp root[n=R*F];
sp bank[n=E*F];
sp sums[n=R*F];
sp out[n=R*Q];
sp inv[n=R];
u32 ids[n=R];
xp target[n=E];
u16 clocks[n=(K+2)*6];
u16 tsc[3] address(CE_TSC_ADDR);
u16 ctrl[1] address(CE_PERF_CNTR_CTRL_ADDR);
inline function stamp(xp p) {
    clocks[p]=tsc[0]; clocks[p+1]=tsc[1]; clocks[p+2]=tsc[2];
}
main {
    ctrl[0]=ctrl[0]|4;
    frame in [0,339) {
        u32 trial=(u32)rx[]; xp parts=(xp)(u32)rx[];
        if (frame==0) { k in [0,(2*F+1)*Q) matrix[k] <- rx[]; }
        xp nrows=0;
        who in [0,P) {
            xp count=(xp)(u32)rx[];
            if (who==X) {
                nrows=count;
                k in [0,count) ids[k] <- (u32)rx[];
                k in [0,count) inv[k] <- rx[];
                k in [0,count*F) root[k] <- rx[];
            } else {
                k in [0,count) bank[k] <- rx[];
                k in [0,count) bank[k] <- rx[];
                k in [0,count*F) bank[k] <- rx[];
            }
        }
        stamp(0);
        k in [0,R*F) sums[k] <- 0.0;
        stamp(3);
        xp total=0;
        part in [0,parts) {
            who in [0,P) {
                xp count=(xp)(u32)rx[];
                xp nr=(xp)(u32)rx[]; k in [0,nr) target[k] <- (xp)(u32)rx[];
                k in [0,count*F) bank[k] <- rx[];
                if (who==X) {
                    total+=count;
                    stamp(6+6*part);
                    xp edge=0;
                    row in [0,nr) {
                        xp dst=row*F;
                        xp end=target[row];
                        while (edge<end) {
                            ch in [0,F) sums[dst+ch] <- sums[dst+ch]+bank[edge*F+ch];
                            edge+=1;
                        }
                    }
                    stamp(9+6*part);
                }
            }
        }
        stamp(6+6*K);
        row in [0,nrows) {
            xp to=row*Q;
            xp from=row*F;
            o in [0,Q) out[to+o] <- matrix[2*F*Q+o];
            ch in [0,F) {
                sp x=root[from+ch];
                sp m=sums[from+ch]*inv[row];
                if (x!=0.0) { o in [0,Q) out[to+o] <+- matrix[ch*Q+o]*x; }
                if (m!=0.0) { o in [0,Q) out[to+o] <+- matrix[F*Q+ch*Q+o]*m; }
            }
            if (RELU!=0) { o in [0,Q) { if (out[to+o]<0.0) out[to+o]=0.0; } }
        }
        stamp(9+6*K);
        tx[] <- trial; tx[] <- (u32)X; tx[] <- (u32)nrows; tx[] <- (u32)total;
        k in [0,(K+2)*6) tx[] <- (u32)clocks[k];
        k in [0,R) tx[] <- ids[k];
        k in [0,R*Q) tx[] <- out[k];
    }
}
