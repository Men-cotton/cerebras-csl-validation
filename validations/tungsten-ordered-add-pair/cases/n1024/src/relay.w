xp param N;
socket rx [[iq=0]];
socket tx;
block in [0,4) {
    turn in [0,4) {
        word in [0,N+10) tx[] <- (u32)rx[];
    }
}
