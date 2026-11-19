// E149 stable edge order and four temporal feature lanes per physical column.
import arch_hdr;
xp param X;
xp param Y;
socket host_in [[iq=0]];
socket from_n [[iq=1]];
socket from_s [[iq=2]];
socket extra_in [[iq=3]];
socket down;
socket up;
socket extra_out;
socket host_out;
sp weights[n=64];
u32 sources[n=64];
u16 destinations[n=64];
sp features[n=64];
sp partial[n=256];
sp owner[n=256];
u32 packet[n=586];
u16 tsc[3] address(CE_TSC_ADDR);
u16 ctrl[1] address(CE_PERF_CNTR_CTRL_ADDR);
u16 clocks[n=6];

main {
    ctrl[0] = ctrl[0] | 4;
    frame in [0,3) {
        u32 trial = (u32)host_in[];
        u32 tag = (u32)host_in[];
        block in [0,8) {
            word in [0,64) {
                sp value = host_in[];
                if (block == Y*2+X) features[word] = value;
            }
        }
        clocks[0] = tsc[0];
        clocks[1] = tsc[1];
        clocks[2] = tsc[2];
        i in [0,256) partial[i] <- 0.0;
        i in [0,256) owner[i] <- 0.0;
        xp e = 0;
        while (e < 64) {
            xp src = sources[e];
            xp dst = destinations[e];
            sp weight = weights[e];
            i in [0,4) partial[dst+i] <+- features[src+i]*weight;
            e += 1;
        }
        if (Y == 0) {
            i in [0,256) down[] <- partial[i];
        } else if (Y < 3) {
            i in [0,256) down[] <- from_n[] + partial[i];
        } else {
            i in [0,256) owner[i] <- from_n[] + partial[i];
        }
        clocks[3] = tsc[0];
        clocks[4] = tsc[1];
        clocks[5] = tsc[2];
        // The host checks every PE header, its clocks and every final owner value.
        if (tag != 0x53454c4c:u32) trial = 0:u32;
        packet[0] = trial;
        packet[1] = (u32)(2*Y+X);
        packet[2] = (u32)e;
        i in [0,6) packet[3+i] <- (u32)clocks[i];
        if (Y > 0) {
            i in [0,9) up[] <- packet[i];
            if (Y == 3) {
                i in [0,256) up[] <- owner[i];
            } else {
                i in [0,(3-Y)*9+256) up[] <- (u32)from_s[];
            }
            if ((Y == 1) && (X == 0)) {
                i in [0,584) host_out[] <- (u32)extra_in[];
            }
        } else {
            i in [0,283) packet[9+i] <- (u32)from_s[];
            if (X == 0) {
                i in [0,292) packet[292+i] <- (u32)extra_in[];
                i in [0,584) extra_out[] <- packet[i];
            } else {
                i in [0,292) extra_out[] <- packet[i];
            }
        }
    }
}
