import arch_hdr;
xp param X;
xp param P;
xp param R;
xp param D;
xp param F;
xp param Q;
xp param E;
xp param K;
xp param S;
socket left [[iq=1]];
socket right;
socket rx [[iq=0]];
socket tx;
sp matrix[n=D*2*F+F+(2*F+1)*Q] bank(2);
sp root[n=R*F];
sp bank[n=E*F] bank(0);
sp sums[n=R*F];
sp out[n=R*Q];
sp inv[n=R];
u32 ids[n=R];
xp ends[n=R];
u16 clocks[n=(K+3)*6];
u16 tsc[3] address(CE_TSC_ADDR);
u16 ctrl[1] address(CE_PERF_CNTR_CTRL_ADDR);
inline function stamp(xp i) {
    clocks[i]=tsc[0]; clocks[i+1]=tsc[1]; clocks[i+2]=tsc[2];
}
main {
    ctrl[0]=ctrl[0]|4;
    xp frames=(xp)(u32)rx[];
    tx[] <- (u32)frames;
    ch in [0,D) { o in [0,2*F) matrix[o*D+ch] <- rx[]; }
    m in [0,F+(2*F+1)*Q) matrix[D*2*F+m] <- rx[];
    frame in [0,frames) {
        u32 tag=(u32)rx[];
        xp encoded=(xp)(u32)rx[];
        xp sharded=(encoded>=4) ? 1 : 0;
        xp mode=encoded&3;
        xp parts=(xp)(u32)rx[];
        xp nr=0;
        who in [0,P) {
            xp n=(xp)(u32)rx[];
            if (who==X) {
                nr=n;
                i in [0,n) ids[i] <- (u32)rx[];
                if (mode==0) { i in [0,n*D) bank[i] <- rx[]; }
                else {
                    i in [0,n) inv[i] <- rx[];
                    i in [0,n*F) root[i] <- rx[];
                }
            } else {
                i in [0,n) out[0] <- rx[];
                if (mode==0) { i in [0,n*D) out[0] <- rx[]; }
                else {
                    i in [0,n) bank[i] <- rx[];
                    i in [0,n*F) bank[i] <- rx[];
                }
            }
        }
        stamp(0);
        if (mode==0) { if (nr<R) { row in [nr,R) { o in [0,2*F) out[row*Q+o] <- 0.0; } } }
        else { i in [0,R*F) sums[i] <- 0.0; }
        stamp(3);
        xp total=0;
        part in [0,parts) {
            who in [0,P) {
                xp count=(xp)(u32)rx[];
                xp n=(xp)(u32)rx[];
                j in [0,n) ends[j] <- (xp)(u32)rx[];
                j in [0,count*F) bank[j] <- rx[];
                if (who==X) {
                    total+=count;
                    stamp(6+6*part);
                    xp edge=0;
                    row in [0,n) {
                        xp to=row*F;
                        xp last=ends[row];
                        while (edge<last) {
                            i in [0,F) sums[to+i] <- sums[to+i]+bank[edge*F+i];
                            edge+=1;
                        }
                    }
                    stamp(9+6*part);
                }
            }
        }
        xp off=D*2*F+F;
        stamp(6+6*parts);
        if (sharded) {
            if (X%S==0) { i in [0,nr*F) right[] <- sums[i]; }
            else if (X%S<S-1) { i in [0,nr*F) right[] <- left[]+sums[i]; }
            else { i in [0,nr*F) sums[i] <- left[]+sums[i]; }
            stamp(9+6*parts);stamp(12+6*parts);
        }
        if (!sharded || X%S==S-1) {
        row in [0,nr) {
            xp dst=row*Q;
            if (mode==0) {
                [[dot.id(0)]]
                out[dst+0] <- ∑ ch in [0,D) bank[row*D+ch]*matrix[0*D+ch];
                [[dot.id(1)]]
                out[dst+1] <- ∑ ch in [0,D) bank[row*D+ch]*matrix[1*D+ch];
                [[dot.id(2)]]
                out[dst+2] <- ∑ ch in [0,D) bank[row*D+ch]*matrix[2*D+ch];
                [[dot.id(3)]]
                out[dst+3] <- ∑ ch in [0,D) bank[row*D+ch]*matrix[3*D+ch];
                [[dot.id(0)]]
                out[dst+4] <- ∑ ch in [0,D) bank[row*D+ch]*matrix[4*D+ch];
                [[dot.id(1)]]
                out[dst+5] <- ∑ ch in [0,D) bank[row*D+ch]*matrix[5*D+ch];
                [[dot.id(2)]]
                out[dst+6] <- ∑ ch in [0,D) bank[row*D+ch]*matrix[6*D+ch];
                [[dot.id(3)]]
                out[dst+7] <- ∑ ch in [0,D) bank[row*D+ch]*matrix[7*D+ch];
                [[dot.id(0)]]
                out[dst+8] <- ∑ ch in [0,D) bank[row*D+ch]*matrix[8*D+ch];
                [[dot.id(1)]]
                out[dst+9] <- ∑ ch in [0,D) bank[row*D+ch]*matrix[9*D+ch];
                [[dot.id(2)]]
                out[dst+10] <- ∑ ch in [0,D) bank[row*D+ch]*matrix[10*D+ch];
                [[dot.id(3)]]
                out[dst+11] <- ∑ ch in [0,D) bank[row*D+ch]*matrix[11*D+ch];
                [[dot.id(0)]]
                out[dst+12] <- ∑ ch in [0,D) bank[row*D+ch]*matrix[12*D+ch];
                [[dot.id(1)]]
                out[dst+13] <- ∑ ch in [0,D) bank[row*D+ch]*matrix[13*D+ch];
                [[dot.id(2)]]
                out[dst+14] <- ∑ ch in [0,D) bank[row*D+ch]*matrix[14*D+ch];
                [[dot.id(3)]]
                out[dst+15] <- ∑ ch in [0,D) bank[row*D+ch]*matrix[15*D+ch];
                [[dot.id(0)]]
                out[dst+16] <- ∑ ch in [0,D) bank[row*D+ch]*matrix[16*D+ch];
                [[dot.id(1)]]
                out[dst+17] <- ∑ ch in [0,D) bank[row*D+ch]*matrix[17*D+ch];
            } else if (mode==1) {
                o in [0,F) {
                    sp v=(root[row*F+o]+sums[row*F+o]*inv[row])+matrix[D*2*F+o];
                    if (v<0.0) v=0.0;
                    out[dst+o]=v;
                }
            } else {
                o in [0,Q) out[dst+o] <- matrix[off+2*F*Q+o];
                ch in [0,F) {
                    sp a=root[row*F+ch];
                    sp b=sums[row*F+ch]*inv[row];
                    if (a!=0.0) { o in [0,Q) out[dst+o] <+- matrix[off+ch*Q+o]*a; }
                    if (b!=0.0) { o in [0,Q) out[dst+o] <+- matrix[off+F*Q+ch*Q+o]*b; }
                }
            }
        }
        }
        stamp(9+6*(parts+sharded));
        xp fields=Q;
        if (mode==0) fields=2*F;
        if (mode==1) fields=F;
        xp words=4+(parts+2+sharded)*6+R*(1+fields);
        tx[] <- (u32)words;
        tx[] <- tag; tx[] <- (u32)X; tx[] <- (u32)((sharded && X%S<S-1) ? 0 : nr); tx[] <- (u32)total;
        i in [0,(parts+2+sharded)*6) tx[] <- (u32)clocks[i];
        i in [0,R) tx[] <- ids[i];
        row in [0,R) { i in [0,fields) tx[] <- out[row*Q+i]; }
    }
}
