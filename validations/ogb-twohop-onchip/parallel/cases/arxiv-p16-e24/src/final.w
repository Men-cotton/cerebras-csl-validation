import arch_hdr;
xp param P,R,H,N,Z,A,Q,FRAMES,SETUP;
socket local_rx [[iq=0]];
socket right_rx [[iq=1]];
socket lower_rx [[iq=2]];
socket tx;
u32 nodes[n=N];
u32 seeds[n=Z];
// Direct Link supplies one 32-bit wavelet per original index.
u32 root[n=Z];
u32 end[n=Z];
u32 source[n=A];
sp inverse[n=Z];
sp weight[n=(2*H+1)*Q];
sp hidden[n=N*H];
u32 bits[n=N*H] address(&hidden);
xp seen[n=N];
u32 packet[n=16+R*(1+H)];
sp numeric[n=16+R*(1+H)] address(&packet);
u32 stats[n=P*16];
sp sum[n=H];
sp out[n=Z*Q];
u16 clocks[n=12];
u16 data_start[n=3];
u16 tsc[3] address(CE_TSC_ADDR);
u16 ctrl[1] address(CE_PERF_CNTR_CTRL_ADDR);
inline function stamp(xp n) {
    clocks[n]=tsc[0]; clocks[n+1]=tsc[1]; clocks[n+2]=tsc[2];
}
main {
    ctrl[0]=ctrl[0]|4;
    i in [0,N) nodes[i] <- (u32)local_rx[];
    i in [0,Z) seeds[i] <- (u32)local_rx[];
    i in [0,Z) root[i] <- (u32)local_rx[];
    i in [0,Z) end[i] <- (u32)local_rx[];
    i in [0,A) source[i] <- (u32)local_rx[];
    i in [0,Z) inverse[i] <- local_rx[];
    i in [0,(2*H+1)*Q) weight[i] <- local_rx[];
    frame in [0,FRAMES) {
        u32 ticket=(u32)local_rx[]; u32 check=(u32)local_rx[];
        stamp(0);
        tx[] <- ticket; tx[] <- check;
        u32 tag=(u32)local_rx[];
        xp echo=(xp)(u32)local_rx[];
        data_start[0]=tsc[0]; data_start[1]=tsc[1]; data_start[2]=tsc[2];
        i in [0,N) seen[i] <- 0;
        u32 errors=0; if (ticket!=tag) errors+=1;
        who in [0,P) {
            if (who==0) { i in [0,16+R*(1+H)) packet[i] <- (u32)local_rx[]; }
            else if (who<8) { i in [0,16+R*(1+H)) packet[i] <- (u32)right_rx[]; }
            else { i in [0,16+R*(1+H)) packet[i] <- (u32)lower_rx[]; }
            i in [0,16) stats[who*16+i] <- packet[i];
            xp rows=(xp)packet[2];
            if (packet[0]!=tag) errors+=1;
            if (packet[1]!=(u32)who) errors+=1;
            row in [0,rows) {
                u32 id=packet[16+row];
                xp lo=0;
                xp hi=N;
                while (lo<hi) {
                    xp mid=lo+(hi-lo)/2;
                    if (nodes[mid]<id) lo=mid+1;
                    else hi=mid;
                }
                if (lo==N) { errors+=1; lo=0; }
                if (nodes[lo]!=id) errors+=1;
                seen[lo]+=1;
                k in [0,H) hidden[lo*H+k] <- numeric[16+R+row*H+k];
            }
        }
        i in [0,N) { if (seen[i]!=1) errors+=1; }
        stamp(3);
        if (echo!=0) {
            i in [0,N*H) tx[] <- bits[i];
            i in [0,N*H) {
                u32 word=(u32)local_rx[];
                if (word!=bits[i]) errors+=1;
                bits[i]=word;
            }
        }
        stamp(6);
        xp edge=0;
        row in [0,Z) {
            k in [0,H) sum[k] <- 0.0;
            while (edge<(xp)end[row]) {
                xp base=(xp)source[edge]*H;
                k in [0,H) sum[k] <- sum[k]+hidden[base+k];
                edge+=1;
            }
            xp to=row*Q;
            k in [0,Q) out[to+k] <- weight[2*H*Q+k];
            ch in [0,H) {
                sp x=hidden[(xp)root[row]*H+ch];
                sp m=sum[ch]*inverse[row];
                k in [0,Q) out[to+k] <+- weight[ch*Q+k]*x;
                k in [0,Q) out[to+k] <+- weight[H*Q+ch*Q+k]*m;
            }
        }
        stamp(9);
        u32 hash=5381;
        i in [0,N*H) { hash=(hash<<5)+hash; hash+=bits[i]; }
        tx[] <- tag; tx[] <- (u32)echo; tx[] <- errors; tx[] <- hash; tx[] <- (u32)edge;
        k in [0,12) tx[] <- (u32)clocks[k];
        k in [0,P*16) tx[] <- stats[k];
        k in [0,Z) tx[] <- seeds[k];
        k in [0,Z*Q) tx[] <- out[k];
        if ((17+P*16+Z*(1+Q))%2!=0) tx[] <- tag;
        tx[] <- tag; k in [0,3) tx[] <- (u32)data_start[k];
    }
}
