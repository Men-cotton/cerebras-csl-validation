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
    frame in [0,3) {
        u32 trial=(u32)rx[];
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
        part in [0,K) {
            who in [0,P) {
                xp count=(xp)(u32)rx[];
                k in [0,count) target[k] <- (xp)(u32)rx[];
                k in [0,count*F) bank[k] <- rx[];
                if (who==X) {
                    total+=count;
                    stamp(6+6*part);
                    edge in [0,count) {
                        xp dst=target[edge]*F;
                        ch in [0,F) sums[dst+ch] <- sums[dst+ch]+bank[edge*F+ch];
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
                sp x0=root[from+0];
                sp m0=sums[from+0]*inv[row];
                if (x0!=0.0) { o in [0,Q) out[to+o] <+- matrix[0*Q+o]*x0; }
                if (m0!=0.0) { o in [0,Q) out[to+o] <+- matrix[F*Q+0*Q+o]*m0; }
                sp x1=root[from+1];
                sp m1=sums[from+1]*inv[row];
                if (x1!=0.0) { o in [0,Q) out[to+o] <+- matrix[1*Q+o]*x1; }
                if (m1!=0.0) { o in [0,Q) out[to+o] <+- matrix[F*Q+1*Q+o]*m1; }
                sp x2=root[from+2];
                sp m2=sums[from+2]*inv[row];
                if (x2!=0.0) { o in [0,Q) out[to+o] <+- matrix[2*Q+o]*x2; }
                if (m2!=0.0) { o in [0,Q) out[to+o] <+- matrix[F*Q+2*Q+o]*m2; }
                sp x3=root[from+3];
                sp m3=sums[from+3]*inv[row];
                if (x3!=0.0) { o in [0,Q) out[to+o] <+- matrix[3*Q+o]*x3; }
                if (m3!=0.0) { o in [0,Q) out[to+o] <+- matrix[F*Q+3*Q+o]*m3; }
                sp x4=root[from+4];
                sp m4=sums[from+4]*inv[row];
                if (x4!=0.0) { o in [0,Q) out[to+o] <+- matrix[4*Q+o]*x4; }
                if (m4!=0.0) { o in [0,Q) out[to+o] <+- matrix[F*Q+4*Q+o]*m4; }
                sp x5=root[from+5];
                sp m5=sums[from+5]*inv[row];
                if (x5!=0.0) { o in [0,Q) out[to+o] <+- matrix[5*Q+o]*x5; }
                if (m5!=0.0) { o in [0,Q) out[to+o] <+- matrix[F*Q+5*Q+o]*m5; }
                sp x6=root[from+6];
                sp m6=sums[from+6]*inv[row];
                if (x6!=0.0) { o in [0,Q) out[to+o] <+- matrix[6*Q+o]*x6; }
                if (m6!=0.0) { o in [0,Q) out[to+o] <+- matrix[F*Q+6*Q+o]*m6; }
                sp x7=root[from+7];
                sp m7=sums[from+7]*inv[row];
                if (x7!=0.0) { o in [0,Q) out[to+o] <+- matrix[7*Q+o]*x7; }
                if (m7!=0.0) { o in [0,Q) out[to+o] <+- matrix[F*Q+7*Q+o]*m7; }
                sp x8=root[from+8];
                sp m8=sums[from+8]*inv[row];
                if (x8!=0.0) { o in [0,Q) out[to+o] <+- matrix[8*Q+o]*x8; }
                if (m8!=0.0) { o in [0,Q) out[to+o] <+- matrix[F*Q+8*Q+o]*m8; }
            if (RELU!=0) { o in [0,Q) { if (out[to+o]<0.0) out[to+o]=0.0; } }
        }
        stamp(9+6*K);
        tx[] <- trial; tx[] <- (u32)X; tx[] <- (u32)nrows; tx[] <- (u32)total;
        k in [0,(K+2)*6) tx[] <- (u32)clocks[k];
        k in [0,R) tx[] <- ids[k];
        k in [0,R*Q) tx[] <- out[k];
    }
}
