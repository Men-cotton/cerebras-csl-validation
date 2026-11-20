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
            if (count >= 16) {
                i in [0,2) partial[(xp)destinations[0]+i] <+- active[(xp)sources[0]+i]*weights[0];
                i in [0,2) partial[(xp)destinations[1]+i] <+- active[(xp)sources[1]+i]*weights[1];
                i in [0,2) partial[(xp)destinations[2]+i] <+- active[(xp)sources[2]+i]*weights[2];
                i in [0,2) partial[(xp)destinations[3]+i] <+- active[(xp)sources[3]+i]*weights[3];
                i in [0,2) partial[(xp)destinations[4]+i] <+- active[(xp)sources[4]+i]*weights[4];
                i in [0,2) partial[(xp)destinations[5]+i] <+- active[(xp)sources[5]+i]*weights[5];
                i in [0,2) partial[(xp)destinations[6]+i] <+- active[(xp)sources[6]+i]*weights[6];
                i in [0,2) partial[(xp)destinations[7]+i] <+- active[(xp)sources[7]+i]*weights[7];
                i in [0,2) partial[(xp)destinations[8]+i] <+- active[(xp)sources[8]+i]*weights[8];
                i in [0,2) partial[(xp)destinations[9]+i] <+- active[(xp)sources[9]+i]*weights[9];
                i in [0,2) partial[(xp)destinations[10]+i] <+- active[(xp)sources[10]+i]*weights[10];
                i in [0,2) partial[(xp)destinations[11]+i] <+- active[(xp)sources[11]+i]*weights[11];
                i in [0,2) partial[(xp)destinations[12]+i] <+- active[(xp)sources[12]+i]*weights[12];
                i in [0,2) partial[(xp)destinations[13]+i] <+- active[(xp)sources[13]+i]*weights[13];
                i in [0,2) partial[(xp)destinations[14]+i] <+- active[(xp)sources[14]+i]*weights[14];
                i in [0,2) partial[(xp)destinations[15]+i] <+- active[(xp)sources[15]+i]*weights[15];
            }
            if (count >= 32) {
                i in [0,2) partial[(xp)destinations[16]+i] <+- active[(xp)sources[16]+i]*weights[16];
                i in [0,2) partial[(xp)destinations[17]+i] <+- active[(xp)sources[17]+i]*weights[17];
                i in [0,2) partial[(xp)destinations[18]+i] <+- active[(xp)sources[18]+i]*weights[18];
                i in [0,2) partial[(xp)destinations[19]+i] <+- active[(xp)sources[19]+i]*weights[19];
                i in [0,2) partial[(xp)destinations[20]+i] <+- active[(xp)sources[20]+i]*weights[20];
                i in [0,2) partial[(xp)destinations[21]+i] <+- active[(xp)sources[21]+i]*weights[21];
                i in [0,2) partial[(xp)destinations[22]+i] <+- active[(xp)sources[22]+i]*weights[22];
                i in [0,2) partial[(xp)destinations[23]+i] <+- active[(xp)sources[23]+i]*weights[23];
                i in [0,2) partial[(xp)destinations[24]+i] <+- active[(xp)sources[24]+i]*weights[24];
                i in [0,2) partial[(xp)destinations[25]+i] <+- active[(xp)sources[25]+i]*weights[25];
                i in [0,2) partial[(xp)destinations[26]+i] <+- active[(xp)sources[26]+i]*weights[26];
                i in [0,2) partial[(xp)destinations[27]+i] <+- active[(xp)sources[27]+i]*weights[27];
                i in [0,2) partial[(xp)destinations[28]+i] <+- active[(xp)sources[28]+i]*weights[28];
                i in [0,2) partial[(xp)destinations[29]+i] <+- active[(xp)sources[29]+i]*weights[29];
                i in [0,2) partial[(xp)destinations[30]+i] <+- active[(xp)sources[30]+i]*weights[30];
                i in [0,2) partial[(xp)destinations[31]+i] <+- active[(xp)sources[31]+i]*weights[31];
            }
            if (count >= 48) {
                i in [0,2) partial[(xp)destinations[32]+i] <+- active[(xp)sources[32]+i]*weights[32];
                i in [0,2) partial[(xp)destinations[33]+i] <+- active[(xp)sources[33]+i]*weights[33];
                i in [0,2) partial[(xp)destinations[34]+i] <+- active[(xp)sources[34]+i]*weights[34];
                i in [0,2) partial[(xp)destinations[35]+i] <+- active[(xp)sources[35]+i]*weights[35];
                i in [0,2) partial[(xp)destinations[36]+i] <+- active[(xp)sources[36]+i]*weights[36];
                i in [0,2) partial[(xp)destinations[37]+i] <+- active[(xp)sources[37]+i]*weights[37];
                i in [0,2) partial[(xp)destinations[38]+i] <+- active[(xp)sources[38]+i]*weights[38];
                i in [0,2) partial[(xp)destinations[39]+i] <+- active[(xp)sources[39]+i]*weights[39];
                i in [0,2) partial[(xp)destinations[40]+i] <+- active[(xp)sources[40]+i]*weights[40];
                i in [0,2) partial[(xp)destinations[41]+i] <+- active[(xp)sources[41]+i]*weights[41];
                i in [0,2) partial[(xp)destinations[42]+i] <+- active[(xp)sources[42]+i]*weights[42];
                i in [0,2) partial[(xp)destinations[43]+i] <+- active[(xp)sources[43]+i]*weights[43];
                i in [0,2) partial[(xp)destinations[44]+i] <+- active[(xp)sources[44]+i]*weights[44];
                i in [0,2) partial[(xp)destinations[45]+i] <+- active[(xp)sources[45]+i]*weights[45];
                i in [0,2) partial[(xp)destinations[46]+i] <+- active[(xp)sources[46]+i]*weights[46];
                i in [0,2) partial[(xp)destinations[47]+i] <+- active[(xp)sources[47]+i]*weights[47];
            }
            if (count >= 64) {
                i in [0,2) partial[(xp)destinations[48]+i] <+- active[(xp)sources[48]+i]*weights[48];
                i in [0,2) partial[(xp)destinations[49]+i] <+- active[(xp)sources[49]+i]*weights[49];
                i in [0,2) partial[(xp)destinations[50]+i] <+- active[(xp)sources[50]+i]*weights[50];
                i in [0,2) partial[(xp)destinations[51]+i] <+- active[(xp)sources[51]+i]*weights[51];
                i in [0,2) partial[(xp)destinations[52]+i] <+- active[(xp)sources[52]+i]*weights[52];
                i in [0,2) partial[(xp)destinations[53]+i] <+- active[(xp)sources[53]+i]*weights[53];
                i in [0,2) partial[(xp)destinations[54]+i] <+- active[(xp)sources[54]+i]*weights[54];
                i in [0,2) partial[(xp)destinations[55]+i] <+- active[(xp)sources[55]+i]*weights[55];
                i in [0,2) partial[(xp)destinations[56]+i] <+- active[(xp)sources[56]+i]*weights[56];
                i in [0,2) partial[(xp)destinations[57]+i] <+- active[(xp)sources[57]+i]*weights[57];
                i in [0,2) partial[(xp)destinations[58]+i] <+- active[(xp)sources[58]+i]*weights[58];
                i in [0,2) partial[(xp)destinations[59]+i] <+- active[(xp)sources[59]+i]*weights[59];
                i in [0,2) partial[(xp)destinations[60]+i] <+- active[(xp)sources[60]+i]*weights[60];
                i in [0,2) partial[(xp)destinations[61]+i] <+- active[(xp)sources[61]+i]*weights[61];
                i in [0,2) partial[(xp)destinations[62]+i] <+- active[(xp)sources[62]+i]*weights[62];
                i in [0,2) partial[(xp)destinations[63]+i] <+- active[(xp)sources[63]+i]*weights[63];
            }
            if (count >= 80) {
                i in [0,2) partial[(xp)destinations[64]+i] <+- active[(xp)sources[64]+i]*weights[64];
                i in [0,2) partial[(xp)destinations[65]+i] <+- active[(xp)sources[65]+i]*weights[65];
                i in [0,2) partial[(xp)destinations[66]+i] <+- active[(xp)sources[66]+i]*weights[66];
                i in [0,2) partial[(xp)destinations[67]+i] <+- active[(xp)sources[67]+i]*weights[67];
                i in [0,2) partial[(xp)destinations[68]+i] <+- active[(xp)sources[68]+i]*weights[68];
                i in [0,2) partial[(xp)destinations[69]+i] <+- active[(xp)sources[69]+i]*weights[69];
                i in [0,2) partial[(xp)destinations[70]+i] <+- active[(xp)sources[70]+i]*weights[70];
                i in [0,2) partial[(xp)destinations[71]+i] <+- active[(xp)sources[71]+i]*weights[71];
                i in [0,2) partial[(xp)destinations[72]+i] <+- active[(xp)sources[72]+i]*weights[72];
                i in [0,2) partial[(xp)destinations[73]+i] <+- active[(xp)sources[73]+i]*weights[73];
                i in [0,2) partial[(xp)destinations[74]+i] <+- active[(xp)sources[74]+i]*weights[74];
                i in [0,2) partial[(xp)destinations[75]+i] <+- active[(xp)sources[75]+i]*weights[75];
                i in [0,2) partial[(xp)destinations[76]+i] <+- active[(xp)sources[76]+i]*weights[76];
                i in [0,2) partial[(xp)destinations[77]+i] <+- active[(xp)sources[77]+i]*weights[77];
                i in [0,2) partial[(xp)destinations[78]+i] <+- active[(xp)sources[78]+i]*weights[78];
                i in [0,2) partial[(xp)destinations[79]+i] <+- active[(xp)sources[79]+i]*weights[79];
            }
            if (count >= 96) {
                i in [0,2) partial[(xp)destinations[80]+i] <+- active[(xp)sources[80]+i]*weights[80];
                i in [0,2) partial[(xp)destinations[81]+i] <+- active[(xp)sources[81]+i]*weights[81];
                i in [0,2) partial[(xp)destinations[82]+i] <+- active[(xp)sources[82]+i]*weights[82];
                i in [0,2) partial[(xp)destinations[83]+i] <+- active[(xp)sources[83]+i]*weights[83];
                i in [0,2) partial[(xp)destinations[84]+i] <+- active[(xp)sources[84]+i]*weights[84];
                i in [0,2) partial[(xp)destinations[85]+i] <+- active[(xp)sources[85]+i]*weights[85];
                i in [0,2) partial[(xp)destinations[86]+i] <+- active[(xp)sources[86]+i]*weights[86];
                i in [0,2) partial[(xp)destinations[87]+i] <+- active[(xp)sources[87]+i]*weights[87];
                i in [0,2) partial[(xp)destinations[88]+i] <+- active[(xp)sources[88]+i]*weights[88];
                i in [0,2) partial[(xp)destinations[89]+i] <+- active[(xp)sources[89]+i]*weights[89];
                i in [0,2) partial[(xp)destinations[90]+i] <+- active[(xp)sources[90]+i]*weights[90];
                i in [0,2) partial[(xp)destinations[91]+i] <+- active[(xp)sources[91]+i]*weights[91];
                i in [0,2) partial[(xp)destinations[92]+i] <+- active[(xp)sources[92]+i]*weights[92];
                i in [0,2) partial[(xp)destinations[93]+i] <+- active[(xp)sources[93]+i]*weights[93];
                i in [0,2) partial[(xp)destinations[94]+i] <+- active[(xp)sources[94]+i]*weights[94];
                i in [0,2) partial[(xp)destinations[95]+i] <+- active[(xp)sources[95]+i]*weights[95];
            }
            if (count >= 112) {
                i in [0,2) partial[(xp)destinations[96]+i] <+- active[(xp)sources[96]+i]*weights[96];
                i in [0,2) partial[(xp)destinations[97]+i] <+- active[(xp)sources[97]+i]*weights[97];
                i in [0,2) partial[(xp)destinations[98]+i] <+- active[(xp)sources[98]+i]*weights[98];
                i in [0,2) partial[(xp)destinations[99]+i] <+- active[(xp)sources[99]+i]*weights[99];
                i in [0,2) partial[(xp)destinations[100]+i] <+- active[(xp)sources[100]+i]*weights[100];
                i in [0,2) partial[(xp)destinations[101]+i] <+- active[(xp)sources[101]+i]*weights[101];
                i in [0,2) partial[(xp)destinations[102]+i] <+- active[(xp)sources[102]+i]*weights[102];
                i in [0,2) partial[(xp)destinations[103]+i] <+- active[(xp)sources[103]+i]*weights[103];
                i in [0,2) partial[(xp)destinations[104]+i] <+- active[(xp)sources[104]+i]*weights[104];
                i in [0,2) partial[(xp)destinations[105]+i] <+- active[(xp)sources[105]+i]*weights[105];
                i in [0,2) partial[(xp)destinations[106]+i] <+- active[(xp)sources[106]+i]*weights[106];
                i in [0,2) partial[(xp)destinations[107]+i] <+- active[(xp)sources[107]+i]*weights[107];
                i in [0,2) partial[(xp)destinations[108]+i] <+- active[(xp)sources[108]+i]*weights[108];
                i in [0,2) partial[(xp)destinations[109]+i] <+- active[(xp)sources[109]+i]*weights[109];
                i in [0,2) partial[(xp)destinations[110]+i] <+- active[(xp)sources[110]+i]*weights[110];
                i in [0,2) partial[(xp)destinations[111]+i] <+- active[(xp)sources[111]+i]*weights[111];
            }
            if (count >= 128) {
                i in [0,2) partial[(xp)destinations[112]+i] <+- active[(xp)sources[112]+i]*weights[112];
                i in [0,2) partial[(xp)destinations[113]+i] <+- active[(xp)sources[113]+i]*weights[113];
                i in [0,2) partial[(xp)destinations[114]+i] <+- active[(xp)sources[114]+i]*weights[114];
                i in [0,2) partial[(xp)destinations[115]+i] <+- active[(xp)sources[115]+i]*weights[115];
                i in [0,2) partial[(xp)destinations[116]+i] <+- active[(xp)sources[116]+i]*weights[116];
                i in [0,2) partial[(xp)destinations[117]+i] <+- active[(xp)sources[117]+i]*weights[117];
                i in [0,2) partial[(xp)destinations[118]+i] <+- active[(xp)sources[118]+i]*weights[118];
                i in [0,2) partial[(xp)destinations[119]+i] <+- active[(xp)sources[119]+i]*weights[119];
                i in [0,2) partial[(xp)destinations[120]+i] <+- active[(xp)sources[120]+i]*weights[120];
                i in [0,2) partial[(xp)destinations[121]+i] <+- active[(xp)sources[121]+i]*weights[121];
                i in [0,2) partial[(xp)destinations[122]+i] <+- active[(xp)sources[122]+i]*weights[122];
                i in [0,2) partial[(xp)destinations[123]+i] <+- active[(xp)sources[123]+i]*weights[123];
                i in [0,2) partial[(xp)destinations[124]+i] <+- active[(xp)sources[124]+i]*weights[124];
                i in [0,2) partial[(xp)destinations[125]+i] <+- active[(xp)sources[125]+i]*weights[125];
                i in [0,2) partial[(xp)destinations[126]+i] <+- active[(xp)sources[126]+i]*weights[126];
                i in [0,2) partial[(xp)destinations[127]+i] <+- active[(xp)sources[127]+i]*weights[127];
            }
            if (count >= 144) {
                i in [0,2) partial[(xp)destinations[128]+i] <+- active[(xp)sources[128]+i]*weights[128];
                i in [0,2) partial[(xp)destinations[129]+i] <+- active[(xp)sources[129]+i]*weights[129];
                i in [0,2) partial[(xp)destinations[130]+i] <+- active[(xp)sources[130]+i]*weights[130];
                i in [0,2) partial[(xp)destinations[131]+i] <+- active[(xp)sources[131]+i]*weights[131];
                i in [0,2) partial[(xp)destinations[132]+i] <+- active[(xp)sources[132]+i]*weights[132];
                i in [0,2) partial[(xp)destinations[133]+i] <+- active[(xp)sources[133]+i]*weights[133];
                i in [0,2) partial[(xp)destinations[134]+i] <+- active[(xp)sources[134]+i]*weights[134];
                i in [0,2) partial[(xp)destinations[135]+i] <+- active[(xp)sources[135]+i]*weights[135];
                i in [0,2) partial[(xp)destinations[136]+i] <+- active[(xp)sources[136]+i]*weights[136];
                i in [0,2) partial[(xp)destinations[137]+i] <+- active[(xp)sources[137]+i]*weights[137];
                i in [0,2) partial[(xp)destinations[138]+i] <+- active[(xp)sources[138]+i]*weights[138];
                i in [0,2) partial[(xp)destinations[139]+i] <+- active[(xp)sources[139]+i]*weights[139];
                i in [0,2) partial[(xp)destinations[140]+i] <+- active[(xp)sources[140]+i]*weights[140];
                i in [0,2) partial[(xp)destinations[141]+i] <+- active[(xp)sources[141]+i]*weights[141];
                i in [0,2) partial[(xp)destinations[142]+i] <+- active[(xp)sources[142]+i]*weights[142];
                i in [0,2) partial[(xp)destinations[143]+i] <+- active[(xp)sources[143]+i]*weights[143];
            }
            xp e = (count >> 4) << 4;
            if (e>144) e=144;
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
