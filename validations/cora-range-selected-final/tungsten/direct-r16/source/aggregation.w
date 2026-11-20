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
sp tail_weights[n=170];
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
        xp count = 0; xp residual = 0;
        block in [0,R) {
            u32 encoded = (u32)host_in[]; xp n = (xp)(encoded & 0xffff:u32);
            nread in [0,n) {
                u32 p = (u32)host_in[];
                sp v = host_in[];
                if (block == Y) { pairs[nread] = p; weights[nread] = v; }
            }
            if (block == Y) { count=n; residual=(xp)(encoded >> 16); }
        }
        u32 hg = 0x811c9dc5:u32;
        e in [0,count) { hg = mix(mix(hg,pairs[e]),bitcast(u32,weights[e])); }
        packet[8] = (u32)tsc[0]; packet[9] = (u32)tsc[1]; packet[10] = (u32)tsc[2];
        e in [0,residual) { sources[e]=(u16)(pairs[e] >> 18); destinations[e]=((u16)pairs[e]) >> 2; }
        xp tail=count-residual; xp first_src=0; xp first_dst=0;
        if (tail > 0) {
            first_src=(xp)(pairs[residual] >> 18); first_dst=(xp)(((u16)pairs[residual]) >> 2);
            j in [0,tail) tail_weights[j] <- weights[residual+j];
        }
        packet[11] = (u32)tsc[0]; packet[12] = (u32)tsc[1]; packet[13] = (u32)tsc[2];
        frame in [0,3) {
            u32 trial = (u32)host_in[];
            if (trial == 2:u32) { i in [0,F) active[i] <- bank_b[i]; }
            else { i in [0,F) active[i] <- bank_a[i]; }
            u32 hf = 0x811c9dc5:u32;
            i in [0,F) { hf = mix(hf,bitcast(u32,active[i])); }
            packet[0] = window_id; packet[1] = trial; packet[2] = (u32)(Y*4+X); packet[3] = (u32)count | ((u32)residual << 16);
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
            if (residual >= 16) {
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
            if (residual >= 32) {
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
            if (residual >= 48) {
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
            if (residual >= 64) {
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
            if (residual >= 80) {
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
            if (residual >= 96) {
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
            if (residual >= 112) {
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
            if (residual >= 128) {
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
            if (residual >= 144) {
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
            if (residual >= 160) {
                i in [0,2) partial[(xp)destinations[144]+i] <+- active[(xp)sources[144]+i]*weights[144];
                i in [0,2) partial[(xp)destinations[145]+i] <+- active[(xp)sources[145]+i]*weights[145];
                i in [0,2) partial[(xp)destinations[146]+i] <+- active[(xp)sources[146]+i]*weights[146];
                i in [0,2) partial[(xp)destinations[147]+i] <+- active[(xp)sources[147]+i]*weights[147];
                i in [0,2) partial[(xp)destinations[148]+i] <+- active[(xp)sources[148]+i]*weights[148];
                i in [0,2) partial[(xp)destinations[149]+i] <+- active[(xp)sources[149]+i]*weights[149];
                i in [0,2) partial[(xp)destinations[150]+i] <+- active[(xp)sources[150]+i]*weights[150];
                i in [0,2) partial[(xp)destinations[151]+i] <+- active[(xp)sources[151]+i]*weights[151];
                i in [0,2) partial[(xp)destinations[152]+i] <+- active[(xp)sources[152]+i]*weights[152];
                i in [0,2) partial[(xp)destinations[153]+i] <+- active[(xp)sources[153]+i]*weights[153];
                i in [0,2) partial[(xp)destinations[154]+i] <+- active[(xp)sources[154]+i]*weights[154];
                i in [0,2) partial[(xp)destinations[155]+i] <+- active[(xp)sources[155]+i]*weights[155];
                i in [0,2) partial[(xp)destinations[156]+i] <+- active[(xp)sources[156]+i]*weights[156];
                i in [0,2) partial[(xp)destinations[157]+i] <+- active[(xp)sources[157]+i]*weights[157];
                i in [0,2) partial[(xp)destinations[158]+i] <+- active[(xp)sources[158]+i]*weights[158];
                i in [0,2) partial[(xp)destinations[159]+i] <+- active[(xp)sources[159]+i]*weights[159];
            }
            if (residual >= 176) {
                i in [0,2) partial[(xp)destinations[160]+i] <+- active[(xp)sources[160]+i]*weights[160];
                i in [0,2) partial[(xp)destinations[161]+i] <+- active[(xp)sources[161]+i]*weights[161];
                i in [0,2) partial[(xp)destinations[162]+i] <+- active[(xp)sources[162]+i]*weights[162];
                i in [0,2) partial[(xp)destinations[163]+i] <+- active[(xp)sources[163]+i]*weights[163];
                i in [0,2) partial[(xp)destinations[164]+i] <+- active[(xp)sources[164]+i]*weights[164];
                i in [0,2) partial[(xp)destinations[165]+i] <+- active[(xp)sources[165]+i]*weights[165];
                i in [0,2) partial[(xp)destinations[166]+i] <+- active[(xp)sources[166]+i]*weights[166];
                i in [0,2) partial[(xp)destinations[167]+i] <+- active[(xp)sources[167]+i]*weights[167];
                i in [0,2) partial[(xp)destinations[168]+i] <+- active[(xp)sources[168]+i]*weights[168];
                i in [0,2) partial[(xp)destinations[169]+i] <+- active[(xp)sources[169]+i]*weights[169];
                i in [0,2) partial[(xp)destinations[170]+i] <+- active[(xp)sources[170]+i]*weights[170];
                i in [0,2) partial[(xp)destinations[171]+i] <+- active[(xp)sources[171]+i]*weights[171];
                i in [0,2) partial[(xp)destinations[172]+i] <+- active[(xp)sources[172]+i]*weights[172];
                i in [0,2) partial[(xp)destinations[173]+i] <+- active[(xp)sources[173]+i]*weights[173];
                i in [0,2) partial[(xp)destinations[174]+i] <+- active[(xp)sources[174]+i]*weights[174];
                i in [0,2) partial[(xp)destinations[175]+i] <+- active[(xp)sources[175]+i]*weights[175];
            }
            if (residual >= 192) {
                i in [0,2) partial[(xp)destinations[176]+i] <+- active[(xp)sources[176]+i]*weights[176];
                i in [0,2) partial[(xp)destinations[177]+i] <+- active[(xp)sources[177]+i]*weights[177];
                i in [0,2) partial[(xp)destinations[178]+i] <+- active[(xp)sources[178]+i]*weights[178];
                i in [0,2) partial[(xp)destinations[179]+i] <+- active[(xp)sources[179]+i]*weights[179];
                i in [0,2) partial[(xp)destinations[180]+i] <+- active[(xp)sources[180]+i]*weights[180];
                i in [0,2) partial[(xp)destinations[181]+i] <+- active[(xp)sources[181]+i]*weights[181];
                i in [0,2) partial[(xp)destinations[182]+i] <+- active[(xp)sources[182]+i]*weights[182];
                i in [0,2) partial[(xp)destinations[183]+i] <+- active[(xp)sources[183]+i]*weights[183];
                i in [0,2) partial[(xp)destinations[184]+i] <+- active[(xp)sources[184]+i]*weights[184];
                i in [0,2) partial[(xp)destinations[185]+i] <+- active[(xp)sources[185]+i]*weights[185];
                i in [0,2) partial[(xp)destinations[186]+i] <+- active[(xp)sources[186]+i]*weights[186];
                i in [0,2) partial[(xp)destinations[187]+i] <+- active[(xp)sources[187]+i]*weights[187];
                i in [0,2) partial[(xp)destinations[188]+i] <+- active[(xp)sources[188]+i]*weights[188];
                i in [0,2) partial[(xp)destinations[189]+i] <+- active[(xp)sources[189]+i]*weights[189];
                i in [0,2) partial[(xp)destinations[190]+i] <+- active[(xp)sources[190]+i]*weights[190];
                i in [0,2) partial[(xp)destinations[191]+i] <+- active[(xp)sources[191]+i]*weights[191];
            }
            if (residual >= 208) {
                i in [0,2) partial[(xp)destinations[192]+i] <+- active[(xp)sources[192]+i]*weights[192];
                i in [0,2) partial[(xp)destinations[193]+i] <+- active[(xp)sources[193]+i]*weights[193];
                i in [0,2) partial[(xp)destinations[194]+i] <+- active[(xp)sources[194]+i]*weights[194];
                i in [0,2) partial[(xp)destinations[195]+i] <+- active[(xp)sources[195]+i]*weights[195];
                i in [0,2) partial[(xp)destinations[196]+i] <+- active[(xp)sources[196]+i]*weights[196];
                i in [0,2) partial[(xp)destinations[197]+i] <+- active[(xp)sources[197]+i]*weights[197];
                i in [0,2) partial[(xp)destinations[198]+i] <+- active[(xp)sources[198]+i]*weights[198];
                i in [0,2) partial[(xp)destinations[199]+i] <+- active[(xp)sources[199]+i]*weights[199];
                i in [0,2) partial[(xp)destinations[200]+i] <+- active[(xp)sources[200]+i]*weights[200];
                i in [0,2) partial[(xp)destinations[201]+i] <+- active[(xp)sources[201]+i]*weights[201];
                i in [0,2) partial[(xp)destinations[202]+i] <+- active[(xp)sources[202]+i]*weights[202];
                i in [0,2) partial[(xp)destinations[203]+i] <+- active[(xp)sources[203]+i]*weights[203];
                i in [0,2) partial[(xp)destinations[204]+i] <+- active[(xp)sources[204]+i]*weights[204];
                i in [0,2) partial[(xp)destinations[205]+i] <+- active[(xp)sources[205]+i]*weights[205];
                i in [0,2) partial[(xp)destinations[206]+i] <+- active[(xp)sources[206]+i]*weights[206];
                i in [0,2) partial[(xp)destinations[207]+i] <+- active[(xp)sources[207]+i]*weights[207];
            }
            if (residual >= 224) {
                i in [0,2) partial[(xp)destinations[208]+i] <+- active[(xp)sources[208]+i]*weights[208];
                i in [0,2) partial[(xp)destinations[209]+i] <+- active[(xp)sources[209]+i]*weights[209];
                i in [0,2) partial[(xp)destinations[210]+i] <+- active[(xp)sources[210]+i]*weights[210];
                i in [0,2) partial[(xp)destinations[211]+i] <+- active[(xp)sources[211]+i]*weights[211];
                i in [0,2) partial[(xp)destinations[212]+i] <+- active[(xp)sources[212]+i]*weights[212];
                i in [0,2) partial[(xp)destinations[213]+i] <+- active[(xp)sources[213]+i]*weights[213];
                i in [0,2) partial[(xp)destinations[214]+i] <+- active[(xp)sources[214]+i]*weights[214];
                i in [0,2) partial[(xp)destinations[215]+i] <+- active[(xp)sources[215]+i]*weights[215];
                i in [0,2) partial[(xp)destinations[216]+i] <+- active[(xp)sources[216]+i]*weights[216];
                i in [0,2) partial[(xp)destinations[217]+i] <+- active[(xp)sources[217]+i]*weights[217];
                i in [0,2) partial[(xp)destinations[218]+i] <+- active[(xp)sources[218]+i]*weights[218];
                i in [0,2) partial[(xp)destinations[219]+i] <+- active[(xp)sources[219]+i]*weights[219];
                i in [0,2) partial[(xp)destinations[220]+i] <+- active[(xp)sources[220]+i]*weights[220];
                i in [0,2) partial[(xp)destinations[221]+i] <+- active[(xp)sources[221]+i]*weights[221];
                i in [0,2) partial[(xp)destinations[222]+i] <+- active[(xp)sources[222]+i]*weights[222];
                i in [0,2) partial[(xp)destinations[223]+i] <+- active[(xp)sources[223]+i]*weights[223];
            }
            xp e=(residual >> 4) << 4; if (e>224) e=224;
            while (e<residual) { i in [0,2) partial[(xp)destinations[e]+i] <+- active[(xp)sources[e]+i]*weights[e]; e+=1; }
            if (tail >= 8) {
                i in [0,2) partial[first_dst+0+i] <+- active[first_src+0+i]*tail_weights[0];
                i in [0,2) partial[first_dst+2+i] <+- active[first_src+2+i]*tail_weights[1];
                i in [0,2) partial[first_dst+4+i] <+- active[first_src+4+i]*tail_weights[2];
                i in [0,2) partial[first_dst+6+i] <+- active[first_src+6+i]*tail_weights[3];
                i in [0,2) partial[first_dst+8+i] <+- active[first_src+8+i]*tail_weights[4];
                i in [0,2) partial[first_dst+10+i] <+- active[first_src+10+i]*tail_weights[5];
                i in [0,2) partial[first_dst+12+i] <+- active[first_src+12+i]*tail_weights[6];
                i in [0,2) partial[first_dst+14+i] <+- active[first_src+14+i]*tail_weights[7];
            }
            if (tail >= 16) {
                i in [0,2) partial[first_dst+16+i] <+- active[first_src+16+i]*tail_weights[8];
                i in [0,2) partial[first_dst+18+i] <+- active[first_src+18+i]*tail_weights[9];
                i in [0,2) partial[first_dst+20+i] <+- active[first_src+20+i]*tail_weights[10];
                i in [0,2) partial[first_dst+22+i] <+- active[first_src+22+i]*tail_weights[11];
                i in [0,2) partial[first_dst+24+i] <+- active[first_src+24+i]*tail_weights[12];
                i in [0,2) partial[first_dst+26+i] <+- active[first_src+26+i]*tail_weights[13];
                i in [0,2) partial[first_dst+28+i] <+- active[first_src+28+i]*tail_weights[14];
                i in [0,2) partial[first_dst+30+i] <+- active[first_src+30+i]*tail_weights[15];
            }
            if (tail >= 24) {
                i in [0,2) partial[first_dst+32+i] <+- active[first_src+32+i]*tail_weights[16];
                i in [0,2) partial[first_dst+34+i] <+- active[first_src+34+i]*tail_weights[17];
                i in [0,2) partial[first_dst+36+i] <+- active[first_src+36+i]*tail_weights[18];
                i in [0,2) partial[first_dst+38+i] <+- active[first_src+38+i]*tail_weights[19];
                i in [0,2) partial[first_dst+40+i] <+- active[first_src+40+i]*tail_weights[20];
                i in [0,2) partial[first_dst+42+i] <+- active[first_src+42+i]*tail_weights[21];
                i in [0,2) partial[first_dst+44+i] <+- active[first_src+44+i]*tail_weights[22];
                i in [0,2) partial[first_dst+46+i] <+- active[first_src+46+i]*tail_weights[23];
            }
            if (tail >= 32) {
                i in [0,2) partial[first_dst+48+i] <+- active[first_src+48+i]*tail_weights[24];
                i in [0,2) partial[first_dst+50+i] <+- active[first_src+50+i]*tail_weights[25];
                i in [0,2) partial[first_dst+52+i] <+- active[first_src+52+i]*tail_weights[26];
                i in [0,2) partial[first_dst+54+i] <+- active[first_src+54+i]*tail_weights[27];
                i in [0,2) partial[first_dst+56+i] <+- active[first_src+56+i]*tail_weights[28];
                i in [0,2) partial[first_dst+58+i] <+- active[first_src+58+i]*tail_weights[29];
                i in [0,2) partial[first_dst+60+i] <+- active[first_src+60+i]*tail_weights[30];
                i in [0,2) partial[first_dst+62+i] <+- active[first_src+62+i]*tail_weights[31];
            }
            if (tail >= 40) {
                i in [0,2) partial[first_dst+64+i] <+- active[first_src+64+i]*tail_weights[32];
                i in [0,2) partial[first_dst+66+i] <+- active[first_src+66+i]*tail_weights[33];
                i in [0,2) partial[first_dst+68+i] <+- active[first_src+68+i]*tail_weights[34];
                i in [0,2) partial[first_dst+70+i] <+- active[first_src+70+i]*tail_weights[35];
                i in [0,2) partial[first_dst+72+i] <+- active[first_src+72+i]*tail_weights[36];
                i in [0,2) partial[first_dst+74+i] <+- active[first_src+74+i]*tail_weights[37];
                i in [0,2) partial[first_dst+76+i] <+- active[first_src+76+i]*tail_weights[38];
                i in [0,2) partial[first_dst+78+i] <+- active[first_src+78+i]*tail_weights[39];
            }
            if (tail >= 48) {
                i in [0,2) partial[first_dst+80+i] <+- active[first_src+80+i]*tail_weights[40];
                i in [0,2) partial[first_dst+82+i] <+- active[first_src+82+i]*tail_weights[41];
                i in [0,2) partial[first_dst+84+i] <+- active[first_src+84+i]*tail_weights[42];
                i in [0,2) partial[first_dst+86+i] <+- active[first_src+86+i]*tail_weights[43];
                i in [0,2) partial[first_dst+88+i] <+- active[first_src+88+i]*tail_weights[44];
                i in [0,2) partial[first_dst+90+i] <+- active[first_src+90+i]*tail_weights[45];
                i in [0,2) partial[first_dst+92+i] <+- active[first_src+92+i]*tail_weights[46];
                i in [0,2) partial[first_dst+94+i] <+- active[first_src+94+i]*tail_weights[47];
            }
            if (tail >= 56) {
                i in [0,2) partial[first_dst+96+i] <+- active[first_src+96+i]*tail_weights[48];
                i in [0,2) partial[first_dst+98+i] <+- active[first_src+98+i]*tail_weights[49];
                i in [0,2) partial[first_dst+100+i] <+- active[first_src+100+i]*tail_weights[50];
                i in [0,2) partial[first_dst+102+i] <+- active[first_src+102+i]*tail_weights[51];
                i in [0,2) partial[first_dst+104+i] <+- active[first_src+104+i]*tail_weights[52];
                i in [0,2) partial[first_dst+106+i] <+- active[first_src+106+i]*tail_weights[53];
                i in [0,2) partial[first_dst+108+i] <+- active[first_src+108+i]*tail_weights[54];
                i in [0,2) partial[first_dst+110+i] <+- active[first_src+110+i]*tail_weights[55];
            }
            if (tail >= 64) {
                i in [0,2) partial[first_dst+112+i] <+- active[first_src+112+i]*tail_weights[56];
                i in [0,2) partial[first_dst+114+i] <+- active[first_src+114+i]*tail_weights[57];
                i in [0,2) partial[first_dst+116+i] <+- active[first_src+116+i]*tail_weights[58];
                i in [0,2) partial[first_dst+118+i] <+- active[first_src+118+i]*tail_weights[59];
                i in [0,2) partial[first_dst+120+i] <+- active[first_src+120+i]*tail_weights[60];
                i in [0,2) partial[first_dst+122+i] <+- active[first_src+122+i]*tail_weights[61];
                i in [0,2) partial[first_dst+124+i] <+- active[first_src+124+i]*tail_weights[62];
                i in [0,2) partial[first_dst+126+i] <+- active[first_src+126+i]*tail_weights[63];
            }
            if (tail >= 72) {
                i in [0,2) partial[first_dst+128+i] <+- active[first_src+128+i]*tail_weights[64];
                i in [0,2) partial[first_dst+130+i] <+- active[first_src+130+i]*tail_weights[65];
                i in [0,2) partial[first_dst+132+i] <+- active[first_src+132+i]*tail_weights[66];
                i in [0,2) partial[first_dst+134+i] <+- active[first_src+134+i]*tail_weights[67];
                i in [0,2) partial[first_dst+136+i] <+- active[first_src+136+i]*tail_weights[68];
                i in [0,2) partial[first_dst+138+i] <+- active[first_src+138+i]*tail_weights[69];
                i in [0,2) partial[first_dst+140+i] <+- active[first_src+140+i]*tail_weights[70];
                i in [0,2) partial[first_dst+142+i] <+- active[first_src+142+i]*tail_weights[71];
            }
            if (tail >= 80) {
                i in [0,2) partial[first_dst+144+i] <+- active[first_src+144+i]*tail_weights[72];
                i in [0,2) partial[first_dst+146+i] <+- active[first_src+146+i]*tail_weights[73];
                i in [0,2) partial[first_dst+148+i] <+- active[first_src+148+i]*tail_weights[74];
                i in [0,2) partial[first_dst+150+i] <+- active[first_src+150+i]*tail_weights[75];
                i in [0,2) partial[first_dst+152+i] <+- active[first_src+152+i]*tail_weights[76];
                i in [0,2) partial[first_dst+154+i] <+- active[first_src+154+i]*tail_weights[77];
                i in [0,2) partial[first_dst+156+i] <+- active[first_src+156+i]*tail_weights[78];
                i in [0,2) partial[first_dst+158+i] <+- active[first_src+158+i]*tail_weights[79];
            }
            if (tail >= 88) {
                i in [0,2) partial[first_dst+160+i] <+- active[first_src+160+i]*tail_weights[80];
                i in [0,2) partial[first_dst+162+i] <+- active[first_src+162+i]*tail_weights[81];
                i in [0,2) partial[first_dst+164+i] <+- active[first_src+164+i]*tail_weights[82];
                i in [0,2) partial[first_dst+166+i] <+- active[first_src+166+i]*tail_weights[83];
                i in [0,2) partial[first_dst+168+i] <+- active[first_src+168+i]*tail_weights[84];
                i in [0,2) partial[first_dst+170+i] <+- active[first_src+170+i]*tail_weights[85];
                i in [0,2) partial[first_dst+172+i] <+- active[first_src+172+i]*tail_weights[86];
                i in [0,2) partial[first_dst+174+i] <+- active[first_src+174+i]*tail_weights[87];
            }
            if (tail >= 96) {
                i in [0,2) partial[first_dst+176+i] <+- active[first_src+176+i]*tail_weights[88];
                i in [0,2) partial[first_dst+178+i] <+- active[first_src+178+i]*tail_weights[89];
                i in [0,2) partial[first_dst+180+i] <+- active[first_src+180+i]*tail_weights[90];
                i in [0,2) partial[first_dst+182+i] <+- active[first_src+182+i]*tail_weights[91];
                i in [0,2) partial[first_dst+184+i] <+- active[first_src+184+i]*tail_weights[92];
                i in [0,2) partial[first_dst+186+i] <+- active[first_src+186+i]*tail_weights[93];
                i in [0,2) partial[first_dst+188+i] <+- active[first_src+188+i]*tail_weights[94];
                i in [0,2) partial[first_dst+190+i] <+- active[first_src+190+i]*tail_weights[95];
            }
            if (tail >= 104) {
                i in [0,2) partial[first_dst+192+i] <+- active[first_src+192+i]*tail_weights[96];
                i in [0,2) partial[first_dst+194+i] <+- active[first_src+194+i]*tail_weights[97];
                i in [0,2) partial[first_dst+196+i] <+- active[first_src+196+i]*tail_weights[98];
                i in [0,2) partial[first_dst+198+i] <+- active[first_src+198+i]*tail_weights[99];
                i in [0,2) partial[first_dst+200+i] <+- active[first_src+200+i]*tail_weights[100];
                i in [0,2) partial[first_dst+202+i] <+- active[first_src+202+i]*tail_weights[101];
                i in [0,2) partial[first_dst+204+i] <+- active[first_src+204+i]*tail_weights[102];
                i in [0,2) partial[first_dst+206+i] <+- active[first_src+206+i]*tail_weights[103];
            }
            if (tail >= 112) {
                i in [0,2) partial[first_dst+208+i] <+- active[first_src+208+i]*tail_weights[104];
                i in [0,2) partial[first_dst+210+i] <+- active[first_src+210+i]*tail_weights[105];
                i in [0,2) partial[first_dst+212+i] <+- active[first_src+212+i]*tail_weights[106];
                i in [0,2) partial[first_dst+214+i] <+- active[first_src+214+i]*tail_weights[107];
                i in [0,2) partial[first_dst+216+i] <+- active[first_src+216+i]*tail_weights[108];
                i in [0,2) partial[first_dst+218+i] <+- active[first_src+218+i]*tail_weights[109];
                i in [0,2) partial[first_dst+220+i] <+- active[first_src+220+i]*tail_weights[110];
                i in [0,2) partial[first_dst+222+i] <+- active[first_src+222+i]*tail_weights[111];
            }
            if (tail >= 120) {
                i in [0,2) partial[first_dst+224+i] <+- active[first_src+224+i]*tail_weights[112];
                i in [0,2) partial[first_dst+226+i] <+- active[first_src+226+i]*tail_weights[113];
                i in [0,2) partial[first_dst+228+i] <+- active[first_src+228+i]*tail_weights[114];
                i in [0,2) partial[first_dst+230+i] <+- active[first_src+230+i]*tail_weights[115];
                i in [0,2) partial[first_dst+232+i] <+- active[first_src+232+i]*tail_weights[116];
                i in [0,2) partial[first_dst+234+i] <+- active[first_src+234+i]*tail_weights[117];
                i in [0,2) partial[first_dst+236+i] <+- active[first_src+236+i]*tail_weights[118];
                i in [0,2) partial[first_dst+238+i] <+- active[first_src+238+i]*tail_weights[119];
            }
            if (tail >= 128) {
                i in [0,2) partial[first_dst+240+i] <+- active[first_src+240+i]*tail_weights[120];
                i in [0,2) partial[first_dst+242+i] <+- active[first_src+242+i]*tail_weights[121];
                i in [0,2) partial[first_dst+244+i] <+- active[first_src+244+i]*tail_weights[122];
                i in [0,2) partial[first_dst+246+i] <+- active[first_src+246+i]*tail_weights[123];
                i in [0,2) partial[first_dst+248+i] <+- active[first_src+248+i]*tail_weights[124];
                i in [0,2) partial[first_dst+250+i] <+- active[first_src+250+i]*tail_weights[125];
                i in [0,2) partial[first_dst+252+i] <+- active[first_src+252+i]*tail_weights[126];
                i in [0,2) partial[first_dst+254+i] <+- active[first_src+254+i]*tail_weights[127];
            }
            if (tail >= 136) {
                i in [0,2) partial[first_dst+256+i] <+- active[first_src+256+i]*tail_weights[128];
                i in [0,2) partial[first_dst+258+i] <+- active[first_src+258+i]*tail_weights[129];
                i in [0,2) partial[first_dst+260+i] <+- active[first_src+260+i]*tail_weights[130];
                i in [0,2) partial[first_dst+262+i] <+- active[first_src+262+i]*tail_weights[131];
                i in [0,2) partial[first_dst+264+i] <+- active[first_src+264+i]*tail_weights[132];
                i in [0,2) partial[first_dst+266+i] <+- active[first_src+266+i]*tail_weights[133];
                i in [0,2) partial[first_dst+268+i] <+- active[first_src+268+i]*tail_weights[134];
                i in [0,2) partial[first_dst+270+i] <+- active[first_src+270+i]*tail_weights[135];
            }
            if (tail >= 144) {
                i in [0,2) partial[first_dst+272+i] <+- active[first_src+272+i]*tail_weights[136];
                i in [0,2) partial[first_dst+274+i] <+- active[first_src+274+i]*tail_weights[137];
                i in [0,2) partial[first_dst+276+i] <+- active[first_src+276+i]*tail_weights[138];
                i in [0,2) partial[first_dst+278+i] <+- active[first_src+278+i]*tail_weights[139];
                i in [0,2) partial[first_dst+280+i] <+- active[first_src+280+i]*tail_weights[140];
                i in [0,2) partial[first_dst+282+i] <+- active[first_src+282+i]*tail_weights[141];
                i in [0,2) partial[first_dst+284+i] <+- active[first_src+284+i]*tail_weights[142];
                i in [0,2) partial[first_dst+286+i] <+- active[first_src+286+i]*tail_weights[143];
            }
            if (tail >= 152) {
                i in [0,2) partial[first_dst+288+i] <+- active[first_src+288+i]*tail_weights[144];
                i in [0,2) partial[first_dst+290+i] <+- active[first_src+290+i]*tail_weights[145];
                i in [0,2) partial[first_dst+292+i] <+- active[first_src+292+i]*tail_weights[146];
                i in [0,2) partial[first_dst+294+i] <+- active[first_src+294+i]*tail_weights[147];
                i in [0,2) partial[first_dst+296+i] <+- active[first_src+296+i]*tail_weights[148];
                i in [0,2) partial[first_dst+298+i] <+- active[first_src+298+i]*tail_weights[149];
                i in [0,2) partial[first_dst+300+i] <+- active[first_src+300+i]*tail_weights[150];
                i in [0,2) partial[first_dst+302+i] <+- active[first_src+302+i]*tail_weights[151];
            }
            if (tail >= 160) {
                i in [0,2) partial[first_dst+304+i] <+- active[first_src+304+i]*tail_weights[152];
                i in [0,2) partial[first_dst+306+i] <+- active[first_src+306+i]*tail_weights[153];
                i in [0,2) partial[first_dst+308+i] <+- active[first_src+308+i]*tail_weights[154];
                i in [0,2) partial[first_dst+310+i] <+- active[first_src+310+i]*tail_weights[155];
                i in [0,2) partial[first_dst+312+i] <+- active[first_src+312+i]*tail_weights[156];
                i in [0,2) partial[first_dst+314+i] <+- active[first_src+314+i]*tail_weights[157];
                i in [0,2) partial[first_dst+316+i] <+- active[first_src+316+i]*tail_weights[158];
                i in [0,2) partial[first_dst+318+i] <+- active[first_src+318+i]*tail_weights[159];
            }
            if (tail >= 168) {
                i in [0,2) partial[first_dst+320+i] <+- active[first_src+320+i]*tail_weights[160];
                i in [0,2) partial[first_dst+322+i] <+- active[first_src+322+i]*tail_weights[161];
                i in [0,2) partial[first_dst+324+i] <+- active[first_src+324+i]*tail_weights[162];
                i in [0,2) partial[first_dst+326+i] <+- active[first_src+326+i]*tail_weights[163];
                i in [0,2) partial[first_dst+328+i] <+- active[first_src+328+i]*tail_weights[164];
                i in [0,2) partial[first_dst+330+i] <+- active[first_src+330+i]*tail_weights[165];
                i in [0,2) partial[first_dst+332+i] <+- active[first_src+332+i]*tail_weights[166];
                i in [0,2) partial[first_dst+334+i] <+- active[first_src+334+i]*tail_weights[167];
            }
            xp j=(tail >> 3) << 3;
            while (j<tail) { i in [0,2) partial[first_dst+2*j+i] <+- active[first_src+2*j+i]*tail_weights[j]; j+=1; }
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
