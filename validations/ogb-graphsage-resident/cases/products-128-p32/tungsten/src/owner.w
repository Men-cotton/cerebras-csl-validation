import arch_hdr;
xp param X;
xp param P;
xp param R;
xp param D;
xp param F;
xp param Q;
xp param E;
xp param K;
socket rx [[iq=0]];
socket tx;
sp matrix[n=D*2*F+F+(2*F+1)*Q];
sp rawx[n=R*D];
sp root[n=R*F];
sp bank[n=E*F];
sp sums[n=R*F];
sp out[n=R*Q];
sp inv[n=R];
u32 ids[n=R];
xp ends[n=R];
u16 clocks[n=(K+2)*6];
u16 tsc[3] address(CE_TSC_ADDR);
u16 ctrl[1] address(CE_PERF_CNTR_CTRL_ADDR);
inline function stamp(xp i) {
    clocks[i]=tsc[0]; clocks[i+1]=tsc[1]; clocks[i+2]=tsc[2];
}
main {
    ctrl[0]=ctrl[0]|4;
    xp frames=(xp)(u32)rx[];
    tx[] <- (u32)frames;
    m in [0,D*2*F+F+(2*F+1)*Q) matrix[m] <- rx[];
    frame in [0,frames) {
        u32 tag=(u32)rx[];
        xp mode=(xp)(u32)rx[];
        xp parts=(xp)(u32)rx[];
        xp nr=0;
        who in [0,P) {
            xp n=(xp)(u32)rx[];
            if (who==X) {
                nr=n;
                i in [0,n) ids[i] <- (u32)rx[];
                if (mode==0) { i in [0,n*D) rawx[i] <- rx[]; }
                else {
                    i in [0,n) inv[i] <- rx[];
                    i in [0,n*F) root[i] <- rx[];
                }
            } else {
                i in [0,n) bank[i] <- rx[];
                if (mode==0) { i in [0,n*D) bank[i] <- rx[]; }
                else {
                    i in [0,n) bank[i] <- rx[];
                    i in [0,n*F) bank[i] <- rx[];
                }
            }
        }
        stamp(0);
        if (mode==0) { i in [0,R*Q) out[i] <- 0.0; }
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
        row in [0,nr) {
            xp dst=row*Q;
            if (mode==0) {
                ch in [0,D) {
                    sp v=rawx[row*D+ch];
                    o in [0,2*F) out[dst+o] <+- matrix[ch*2*F+o]*v;
                }
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
        stamp(9+6*parts);
        xp fields=Q;
        if (mode==0) fields=2*F;
        if (mode==1) fields=F;
        xp words=4+(parts+2)*6+R*(1+fields);
        tx[] <- (u32)words;
        tx[] <- tag; tx[] <- (u32)X; tx[] <- (u32)nr; tx[] <- (u32)total;
        i in [0,(parts+2)*6) tx[] <- (u32)clocks[i];
        i in [0,R) tx[] <- ids[i];
        row in [0,R) { i in [0,fields) tx[] <- out[row*Q+i]; }
    }
}
