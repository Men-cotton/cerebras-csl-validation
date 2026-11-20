xp param P,WORDS,FRAMES;
socket local_rx [[iq=0]];
socket right_rx [[iq=1]];
socket fabric;
socket done;
frame in [0,FRAMES) {
    u32 tag=(u32)local_rx[];
    fabric[] <- tag;
    k in [1,WORDS) fabric[] <- (u32)local_rx[];
    k in [0,(P-1)*WORDS) fabric[] <- (u32)right_rx[];
    done[] <- tag; done[] <- (u32)P;
}
