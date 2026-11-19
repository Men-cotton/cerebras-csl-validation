import arch_hdr;
xp param X;
xp param Y;
xp param R;
xp param F;
xp param E;
socket host_in [[iq=0]];
socket from_n [[iq=1]];
socket from_s [[iq=2]];
socket extra_in [[iq=3]];
socket down;
socket up;
socket extra_out;
socket host_out;
u32 pairs[n=E];
sp weights[n=E];
u16 sources[n=E];
u16 destinations[n=E];
sp bank_a[n=F];
sp bank_b[n=F];
sp active[n=F];
sp partial[n=512];
sp owner[n=512];
u32 packet[n=R*20+512];
u16 tsc[3] address(CE_TSC_ADDR);
u16 ctrl[1] address(CE_PERF_CNTR_CTRL_ADDR);
inline function mix(u32 h, u32 value) -> u32 { return ((h << 5) | (h >> 27)) ^ value; }
main {
    block in [0,R*4) {
        i in [0,F) { sp v = host_in[]; if (block == Y*4+X) bank_a[i] = v; }
        i in [0,F) { sp v = host_in[]; if (block == Y*4+X) bank_b[i] = v; }
    }
    u32 ha = 0x811c9dc5:u32;
    u32 hb = 0x811c9dc5:u32;
    i in [0,F) { ha = mix(ha,bitcast(u32,bank_a[i])); hb = mix(hb,bitcast(u32,bank_b[i])); }
    ctrl[0] = ctrl[0] | 4;
    w in [0,11) {
        u32 window_id = (u32)host_in[];
        xp count = 0;
        block in [0,R) {
            xp n = (xp)(u32)host_in[];
            nread in [0,n) {
                u32 p = (u32)host_in[];
                sp v = host_in[];
                if (block == Y) { pairs[nread] = p; weights[nread] = v; }
            }
            if (block == Y) count = n;
        }
        u32 hg = 0x811c9dc5:u32;
        e in [0,count) { hg = mix(mix(hg,pairs[e]),bitcast(u32,weights[e])); }
        packet[8] = (u32)tsc[0]; packet[9] = (u32)tsc[1]; packet[10] = (u32)tsc[2];
        e in [0,count) { sources[e] = (u16)(pairs[e] >> 16); destinations[e] = (u16)pairs[e]; }
        packet[11] = (u32)tsc[0]; packet[12] = (u32)tsc[1]; packet[13] = (u32)tsc[2];
        frame in [0,3) {
            u32 trial = (u32)host_in[];
            if (trial == 2:u32) { i in [0,F) active[i] <- bank_b[i]; }
            else { i in [0,F) active[i] <- bank_a[i]; }
            u32 hf = 0x811c9dc5:u32;
            i in [0,F) { hf = mix(hf,bitcast(u32,active[i])); }
            packet[0] = window_id; packet[1] = trial; packet[2] = (u32)(Y*4+X); packet[3] = (u32)count;
            packet[4] = hg; packet[5] = ha; packet[6] = hb; packet[7] = hf;
            // All source shards in this column have finished graph setup and the panel copy.
            u32 token = trial;
            if (Y < R-1) token = (u32)from_s[];
            if (Y > 0) up[] <- token;
            if (Y > 0) token = (u32)from_n[];
            if (Y < R-1) down[] <- token;
            if (token != trial) packet[1] = 0:u32;
            packet[14] = (u32)tsc[0]; packet[15] = (u32)tsc[1]; packet[16] = (u32)tsc[2];
            i in [0,512) partial[i] <- 0.0;
            xp e = 0;
            while (e < count) {
                xp src = sources[e]; xp dst = destinations[e]; sp v = weights[e];
                i in [0,2) partial[dst+i] <+- active[src+i]*v;
                e += 1;
            }
            if (Y == 0) { i in [0,512) down[] <- partial[i]; }
            else if (Y < R-1) { i in [0,512) down[] <- from_n[] + partial[i]; }
            else { i in [0,512) owner[i] <- from_n[] + partial[i]; }
            packet[17] = (u32)tsc[0]; packet[18] = (u32)tsc[1]; packet[19] = (u32)tsc[2];
            if (Y > 0) {
                i in [0,20) up[] <- packet[i];
                if (Y == R-1) { i in [0,512) up[] <- owner[i]; }
                else { i in [0,(R-1-Y)*20+512) up[] <- (u32)from_s[]; }
                if ((Y == 1) && (X == 0)) { i in [0,4*(R*20+512)) host_out[] <- (u32)extra_in[]; }
            } else {
                i in [0,(R-1)*20+512) packet[20+i] <- (u32)from_s[];
                i in [0,R*20+512) extra_out[] <- packet[i];
                if (X < 3) { i in [0,(3-X)*(R*20+512)) extra_out[] <- (u32)extra_in[]; }
            }
        }
    }
}
