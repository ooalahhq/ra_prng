"""pytest suite for pip-ra-core.

Byte-identity tests shell out to the repo's own CLI binary
(build/bin/ra_core, built by `make build-c` at the repo root) as the
independent reference - this catches the pip binding silently diverging
from the canonical, already-validated ra_core.c formula. Skipped if that
binary hasn't been built.
"""

import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import ra_core

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLI_BIN = _REPO_ROOT / "build" / "bin" / "ra_core"


def _cli_stream(core: str, key: int, addr_hi: int, n: int) -> bytes:
    result = subprocess.run(
        [str(_CLI_BIN), "--stream", core, str(key), str(addr_hi), str(n)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return result.stdout


needs_cli = pytest.mark.skipif(
    not _CLI_BIN.exists(), reason="build/bin/ra_core not built (run `make build-c` at repo root)"
)


@needs_cli
@pytest.mark.parametrize("core,n", [("singleblock", 10), ("singleblock", 255), ("orbit", 700), ("orbit", 10)])
def test_generate_byte_identical_to_cli(core, n):
    key, addr_hi = 0xCAFEBABE, 42
    ref = _cli_stream(core, key, addr_hi, n)
    got = ra_core.generate(key, addr_hi, n, mode=core)
    assert hashlib.md5(got).hexdigest() == hashlib.md5(ref).hexdigest()
    assert got == ref


@needs_cli
def test_engine_next_byte_identical_to_cli_across_multiple_cycles():
    key, addr_hi, n = 0x12345678, 7, 900  # spans 4 orbit cycles (255*3=765 < 900 <= 1020)
    ref = _cli_stream("orbit", key, addr_hi, n)

    e = ra_core.RaCoreEngine(key, addr_hi)
    got = e.next(300) + e.next(400) + e.next(n - 700)
    assert got == ref
    assert e.position == n


def test_generate_auto_mode_matches_forced_mode():
    key, addr_hi = 1, 0
    assert ra_core.generate(key, addr_hi, 10) == ra_core.generate(key, addr_hi, 10, mode="singleblock")
    assert ra_core.generate(key, addr_hi, 700) == ra_core.generate(key, addr_hi, 700, mode="orbit")


def test_generate_singleblock_rejects_over_255():
    with pytest.raises(ValueError):
        ra_core.generate(1, 0, 256, mode="singleblock")


def test_engine_fork_leaves_parent_untouched_and_diverges():
    parent = ra_core.RaCoreEngine(key=555, addr_hi=1)
    parent.next(50)  # advance partway through a cycle first
    child = parent.fork()

    parent_next = parent.next(10)
    child_next = child.next(10)
    assert parent_next != child_next

    # fork() must not have perturbed parent's own trajectory at all: an
    # engine that never forked, given the same 60 words, matches exactly.
    unforked_reference = ra_core.RaCoreEngine(key=555, addr_hi=1)
    unforked_reference.next(50)
    assert unforked_reference.next(10) == parent_next


def test_engine_save_load_roundtrip():
    e1 = ra_core.RaCoreEngine(key=999, addr_hi=1)
    e1.next(130)  # land mid-cycle

    path = "/tmp/ra_core_test_engine_state.bin"
    e1.save(path)
    e2 = ra_core.RaCoreEngine.load(path)

    assert e2.key == e1.key
    assert e2.addr_hi == e1.addr_hi
    assert e2.position == e1.position
    assert e1.next(40) == e2.next(40)


def test_spawn_streams_are_independent():
    streams = ra_core.spawn_streams(key=123, n=4)
    outs = [s.next(20) for s in streams]
    assert len(set(outs)) == 4
    assert [s.addr_hi for s in streams] == [0, 1, 2, 3]


def test_shuffle_is_a_valid_deterministic_permutation():
    n = 600  # > 255, exercises multi-block key-increment chaining
    seq = list(range(n))
    a = ra_core.shuffle(seq, key=0xABCD)
    b = ra_core.shuffle(seq, key=0xABCD)
    assert sorted(a) == seq
    assert a == b


def test_shuffle_small_n():
    assert sorted(ra_core.shuffle([1, 2, 3], key=1)) == [1, 2, 3]
    assert ra_core.shuffle([], key=1) == []
    assert ra_core.shuffle([1], key=1) == [1]


def test_random_key_is_32_bit():
    k = ra_core.random_key()
    assert 0 <= k <= 0xFFFFFFFF


def test_numpy_bitgenerator_matches_engine_output():
    from ra_core import RaCoreBitGenerator

    key, addr_hi = 42, 0
    # random_raw() exposes next_raw() (aliased to next_uint64 in _bitgen.pyx)
    # directly, one call per output value - unlike Generator.integers(),
    # which applies its own bounded-generation algorithm on top and may
    # consume a different number of underlying draws per output.
    bg = RaCoreBitGenerator(key, addr_hi)
    drawn = bg.random_raw(50)

    e = ra_core.RaCoreEngine(key, addr_hi)
    ref_words = np.frombuffer(e.next(100), dtype="<u4").astype(np.uint64)
    ref = (ref_words[1::2] << 32) | ref_words[0::2]
    assert (drawn == ref).all()


def test_numpy_bitgenerator_deterministic():
    from ra_core import RaCoreBitGenerator

    a = np.random.Generator(RaCoreBitGenerator(7)).integers(0, 100, size=30)
    b = np.random.Generator(RaCoreBitGenerator(7)).integers(0, 100, size=30)
    assert (a == b).all()


def test_lazy_k1_and_step_by_step_next_u32_parity():
    key, addr_hi = 0xDEADBEEF, 99
    # 1. Direct register K=1 evaluation matches generate(..., 1)
    k1_word = ra_core.eval_k1(key, addr_hi)
    ref_bytes = ra_core.generate(key, addr_hi, 600, mode="orbit")
    ref_u32 = np.frombuffer(ref_bytes, dtype="<u4")
    assert k1_word == int(ref_u32[0])

    # 2. Pulling 1 word at a time via next_u32() transitions seamlessly from
    #    lazy K=1 (word 0, no L/M array init) -> cycle 0 (words 1..254) ->
    #    post-reseed cycles (words 255..599) and stays 100% bit-exact.
    e = ra_core.RaCoreEngine(key, addr_hi)
    assert e.position == 0
    first = e.next_u32()
    assert first == k1_word
    assert e.position == 1

    rest = [e.next_u32() for _ in range(599)]
    all_drawn = np.array([first] + rest, dtype="<u4")
    assert (all_drawn == ref_u32).all()
    assert e.position == 600


def test_generate_x8_matches_scalar_singleblock():
    keys = [0, 1, 2, 0xCAFEBABE, 0xDEADBEEF, 0x12345678, 0xFFFFFFFF, 42]
    addr_his = [0, 7, 13, 42, 99, 1024, 0xFFFFFFFF, 5]
    for n in (1, 2, 3, 4, 8, 10, 254, 255):
        interleaved = np.frombuffer(
            ra_core.generate_x8(keys, addr_his, n), dtype="<u4"
        ).reshape(n, 8)
        for lane in range(8):
            ref = np.frombuffer(
                ra_core.generate(keys[lane], addr_his[lane], n, mode="singleblock"),
                dtype="<u4",
            )
            assert (interleaved[:, lane] == ref).all()


def test_golden_kat_v2_final_vectors():
    """Frozen Golden KAT vectors for ra_core v2 Final (W2_D2_xag_opt + Opsi_A3 + eng_v2_E2)."""
    kat_cases = [
        (
            0x00000000,
            0x00000000,
            [
                0xC1DBBB12, 0xFBEDE3B9, 0x93B1E2C8, 0x2067AC16,
                0xC477241F, 0x982A158D, 0xD77DD2E4, 0x89763004,
            ],
            (0x9297471C, 0xEB5F03E7, 0x8970AD1B),
        ),
        (
            0x0000002A,
            0x00000000,
            [
                0x9533450C, 0x288D3BD2, 0x7E833E19, 0xFF96B720,
                0x8CD7BE6B, 0xE91267F5, 0xACC46C12, 0xE7271FF4,
            ],
            (0xB565EF5B, 0x50C62F0B, 0xF7C65DA2),
        ),
    ]
    for key, addr_hi, want_first8, want_boundary in kat_cases:
        assert ra_core.eval_k1(key, addr_hi) == want_first8[0]
        words = np.frombuffer(
            ra_core.generate(key, addr_hi, 300, mode="orbit"), dtype="<u4"
        )
        assert [int(x) for x in words[:8]] == want_first8
        assert (int(words[254]), int(words[255]), int(words[256])) == want_boundary

        eng = ra_core.RaCoreEngine(key, addr_hi)
        eng_words = [eng.next_u32() for _ in range(300)]
        assert eng_words[:8] == want_first8
        assert (eng_words[254], eng_words[255], eng_words[256]) == want_boundary
        assert eng.position == 300
