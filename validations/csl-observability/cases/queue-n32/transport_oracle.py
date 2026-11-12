"""Expected transport results for H099/E178; no protocol simulator or SDK import."""
POISON = 0xDEADBEEF
LEFT = 0x1234ABCD
RIGHT = 0x7890FEDC
SEQUENCE = (0, 1, 0)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def params(mode, n, batch):
    need(type(mode) is int and mode in (0, 1), "mode must be preselected0/range1")
    need(type(n) is int and n in (4, 32), "N must be4/32")
    need(type(batch) is int and batch in (0, 1), "BATCH must be scalar0/batched1")
    return {"MODE": mode, "N": n, "BATCH": batch}


def canonical(n, case):
    params(0, n, 0)
    need(type(case) is int and case in (0, 1), "fixture")
    words = []
    for group in range(n):
        for index in range(6):
            low = (0x3400 + 256*case + 2*group+index if index < 2
                   else 0x3C00 + 4*group+index-2)
            words.append((index << 16) | low)
    return words


def selected(words):
    """Keep complete accepted u32 carriers, including their upper indices."""
    need(all(type(x) is int and 0 <= x < 2**32 for x in words), "wire words")
    return [word for word in words if (word >> 16) <= 1]


def guarded(length):
    return [LEFT] + [POISON]*(length-2) + [RIGHT]


def fixture(mode, n, invocation, batch):
    params(mode, n, batch)
    need(type(invocation) is int and 0 <= invocation < 3, "invocation")
    tag, case = invocation+1, SEQUENCE[invocation]
    stream = canonical(n, case)
    if mode == 0:
        stream = selected(stream)
    stream.append((1 << 16) | (0x6000+tag))
    source = [guarded(6*n+3), guarded(6*n+3)]
    source[0][1:1+len(stream)] = stream
    return {"source": source, "received": [guarded(2*n+3), guarded(2*n+3)],
            "control": [[tag, case], [tag, case]],
            "stats": [[POISON]*32 for _ in range(2)],
            "clocks": [[POISON]*12 for _ in range(2)],
            "scratch_mirror": [[POISON]*(2*n+3) for _ in range(2)]}


def expected_stats(mode, n, invocation, role, batch):
    params(mode, n, batch)
    tag = invocation+1
    result = [mode,n,role,tag,tag,0,0,0,0,0,0,1,0,0,2*n,0,0,0]
    result += [batch,0,0,0,0,2*n+3,0,0,0,0,0,0,0,0]
    if role == 0:
        result[6:9] = [1,1,1]
        result[13] = (2*n if mode == 0 else 6*n)+1
    else:
        result[5] = 2*n
        result[9:11] = [1,1]
        result[15:18] = [2,1,0x6000+tag]
        k = 2*n+1
        result[19:32] = [1,k,1,k,k+2,k,k if batch == 0 else 0,batch,1,2,k,k,k]
    return result


def decode_clocks(words):
    need(len(words) == 12 and all(type(x) is int and 0 <= x <= 65535 for x in words),
         "twelve zero-extended u16 timestamp words")
    stamps = [sum(words[j+i] << (16*i) for i in range(3)) for j in range(0,12,3)]
    start, tx, ack, end = stamps
    need(0 <= start <= tx <= end and start <= ack <= end and end > start,
         "TX/terminal-ACK must both precede joined end; no wrap admitted")
    return dict(start=start, tx=tx, terminal_ack=ack, end=end, elapsed=end-start)


def audit(raw, mode, n, invocation, batch):
    expected = fixture(mode, n, invocation, batch)
    need(set(raw) == set(expected), "complete raw symbol set")
    for name, value in raw.items():
        shape = (2, len(expected[name][0]))
        need(len(value) == shape[0] and all(len(row) == shape[1] for row in value),
             "complete raw shape: "+name)
        need(all(type(x) is int and 0 <= x < 2**32 for row in value for x in row),
             "u32 raw words: "+name)
    for name in ("source", "control"):
        need(raw[name] == expected[name], "resident input or guard changed: "+name)
    words = selected(canonical(n, SEQUENCE[invocation]))
    words.append((1 << 16) | (0x6001+invocation))
    expected_output = expected["received"]
    expected_output[1][1:-1] = [word & 0xFFFF for word in words]
    need(raw["received"] == expected_output, "complete low16 output, poison and guards")
    need(raw["scratch_mirror"] == [guarded(2*n+3), [LEFT]+words+[RIGHT]],
         "complete physical u32 scratch: upper index, payload, terminal, guards and source poison")
    need(raw["stats"] == [expected_stats(mode,n,invocation,r,batch) for r in range(2)],
         "tag/count/callback/terminal/join/index/mask witness")
    need(raw["clocks"][1] == [0]*12, "receiver clock ABI")
    return decode_clocks(raw["clocks"][0])


def costs(mode, n, batch):
    params(mode, n, batch)
    sent = (2*n if mode == 0 else 6*n)+1
    k, length = 2*n+1, 2*n+3
    exported = 4*((6*n+3)+(2*n+3)+2+32+12+length)
    return dict(exported_array_bytes_per_pe=exported,
                h2d_bytes_per_invocation=2*exported, d2h_bytes_per_invocation=2*exported,
                h2d_bytes_per_capture=6*exported, d2h_bytes_per_capture=6*exported,
                data_link_bytes=4*sent, total_directed_link_bytes=4*(sent+2),
                measured_interval_link_bytes=4*(sent+1), ready_bytes_outside_interval=4,
                ready_scope="READY precedes sender start; its TX callback may overlap receiver service",
                host_preselection=dict(canonical_scan_words=6*n if mode == 0 else 0,
                    index_tests=6*n if mode == 0 else 0,
                    selected_copy_words=2*n if mode == 0 else 0,
                    peak_canonical_plus_selected_bytes=4*(8*n if mode == 0 else 6*n),
                    terminal_append_words=1, common_source_words=6*n,
                    scope="logical typed buffers before common guarded packing; Python objects and device timing excluded"),
                useful_words=2*n, rejected_words=0 if mode == 0 else 4*n,
                terminal_words=1, data_link_words=sent, ready_words=1, ack_words=1,
                total_directed_link_words=sent+2, trailing_reject_words=0 if mode == 0 else 4,
                payload_bits_per_word=16, physical_wavelet_bits=32,
                receiver_service=dict(accepted_words=k,
                    physical_scratch_bytes_per_pe=4*length, mirror_bytes_per_pe=4*length,
                    prepare_scratch_stores_per_pe=length+2,
                    receive_scratch_stores=k,
                    receive_scalar_task_entries=k if batch == 0 else 0,
                    receive_batch_posts=batch, rx_done_callbacks=1,
                    receive_counter_increments=3*k if batch == 0 else 0,
                    receive_counter_assignments=3 if batch == 1 else 0,
                    receive_counter_scope="scalar stats25/20/29 per word; batch stats26 post and stats20/29 completed extent; common RXposted excluded",
                    payload_terminal_checks=k, upper_index_checks=k, low16_masks=k,
                    index_mask_counter_increments=2*k, guard_checks=2,
                    final_mirror_words_per_pe=length, final_mirror_words_total=2*length,
                    received_output_stores=k,
                    timed_scratch_read_words=2*length+k+2, timed_scratch_write_words=k,
                    timed_u32_data_store_words=2*length+k,
                    batch_count_scope="completed DSD extent, not independently measured per-word events",
                    final_mirror_counter_increments=2*length,
                    scope="logical source operations, not instruction counts; data counts exclude stats/clocks/state; both mirrors, masks and index validation inside joined interval; prepare outside"),
                arrays_bytes_per_pe=exported+4*length+6+4,
                arrays_scope="exported arrays plus physical u32 scratch and stamp/token; scalar state, descriptors, code and stack excluded")
