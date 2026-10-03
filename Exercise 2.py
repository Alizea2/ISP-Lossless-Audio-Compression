#!/usr/bin/env python
# coding: utf-8

# # EXERCISE 2

# ## This notebook builds a complete lossless audio compression system from scratch using Rice coding and fixed prediction. It first creates custom bit-level tools to write and read individual bits efficiently. The audio samples are transformed using ZigZag encoding so negative values can be handled properly, then compressed using Rice coding with different K parameters. Fixed linear predictors (orders 0–4) are tested to find the one that produces the smallest residual errors. The WAV files are encoded into a custom .ex2 format, decoded back to WAV, and verified to be identical to the original. Finally, compression results and histograms are shown to analyze performance.

# # Imports

# In[1]:


#importing libraries
import os
import struct
import wave
from dataclasses import dataclass
import numpy as np
import time
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt


# # Bit-Level IO: Unary + Bit Reader/Writer

# In[2]:


class BitWriter:
    def __init__(self):
        # buffer to store final bytes
        self.buf = bytearray()
        # current byte being filled
        self.cur = 0
        # how many bits are currently in `cur`
        self.nbits = 0
        # total bits written (for header info)
        self.bit_count = 0

    def _flush_full_byte(self):
        # if we collected 8 bits, push the byte into buffer
        if self.nbits == 8:
            self.buf.append(self.cur & 0xFF)
            self.cur = 0
            self.nbits = 0

    def write_bit(self, b):
        # write a single bit (0/1)
        b = 1 if b else 0
        self.cur = (self.cur << 1) | b
        self.nbits += 1
        self.bit_count += 1
        self._flush_full_byte()

    def write_bits(self, value, n):
        # write n bits from MSB to LSB
        for i in range(n - 1, -1, -1):
            self.write_bit((value >> i) & 1)

    def write_unary_ones_then_zero(self, q):
        # unary code: q times '1' then a final '0'
        if q < 0:
            raise ValueError("q must be >= 0")

        # if current byte is not aligned, fill remaining space with 1s
        if self.nbits != 0 and q > 0:
            space = 8 - self.nbits
            k = space if q >= space else q
            ones = (1 << k) - 1
            self.cur = (self.cur << k) | ones
            self.nbits += k
            self.bit_count += k
            q -= k
            self._flush_full_byte()

        # now we are byte-aligned, so we can add full 0xFF bytes quickly
        if q >= 8:
            full = q // 8
            self.buf.extend(b"\xFF" * full)
            self.bit_count += full * 8
            q -= full * 8

        # write the remaining 1s (< 8)
        if q > 0:
            ones = (1 << q) - 1
            self.cur = (self.cur << q) | ones
            self.nbits += q
            self.bit_count += q
            self._flush_full_byte()

        # terminator 0 (ends unary)
        self.write_bit(0)

    def to_bytes_and_pad(self):
        # pad last byte with 0s if it is not full
        if self.nbits:
            self.cur <<= (8 - self.nbits)
            self.buf.append(self.cur & 0xFF)
            self.cur = 0
            self.nbits = 0
        return bytes(self.buf)


class BitReader:
    _lead1 = None  # lookup table: number of leading 1s for byte 0..255

    def __init__(self, data, total_bits):
        self.data = data
        # total valid bits in the stream (ignore padding bits)
        self.total_bits = int(total_bits)
        # current bit position
        self.pos = 0

        # build lookup table once (for faster unary decoding)
        if BitReader._lead1 is None:
            BitReader._lead1 = [0] * 256
            for b in range(256):
                c = 0
                for i in range(8):
                    if b & (1 << (7 - i)):
                        c += 1
                    else:
                        break
                BitReader._lead1[b] = c

    def read_bit(self):
        # read one bit from stream
        if self.pos >= self.total_bits:
            raise EOFError("Reached end of bit stream")
        byte_i = self.pos >> 3
        bit_i = 7 - (self.pos & 7)
        self.pos += 1
        return (self.data[byte_i] >> bit_i) & 1

    def read_bits(self, n):
        # read n bits and combine into an integer
        v = 0
        for _ in range(n):
            v = (v << 1) | self.read_bit()
        return v

    def read_unary_q(self):
        # unary decoding: count 1s until the first 0
        if self.pos >= self.total_bits:
            raise EOFError("Reached end of bit stream")

        q = 0
        while True:
            if self.pos >= self.total_bits:
                raise EOFError("Unary ran past end of stream")

            byte_i = self.pos >> 3
            offset = self.pos & 7
            b = self.data[byte_i]

            # shift so current position becomes the MSB
            shifted = (b << offset) & 0xFF
            avail = 8 - offset

            # count how many leading 1s in this byte part
            lead = BitReader._lead1[shifted]
            if lead >= avail:
                q += avail
                self.pos += avail
                continue

            q += lead
            self.pos += lead

            # consume the terminating 0 bit
            if self.pos >= self.total_bits:
                raise EOFError("Missing unary terminator 0")
            self.pos += 1
            return q


# # ZigZag Conversion

# In[3]:


#signed to unsigned
def signed_to_zigzag(s):
    s = s.astype(np.int64)
    u = np.where(s >= 0, 2 * s, -2 * s - 1)
    return u.astype(np.uint64)

#nnsigned to signed
def zigzag_to_signed(u):
    u = u.astype(np.uint64)
    s = np.where((u & 1) == 0, (u >> 1), -((u >> 1) + 1))
    return s.astype(np.int64)


# # Rice Coding: Encode/Decode

# In[4]:


def rice_encode_unsigned(values_u, K):
    K = int(K)
    # mask to get the remainder bits (lower K bits)
    mask = (1 << K) - 1 if K > 0 else 0
    bw = BitWriter()  # we write bits into this

    for x in values_u:
        x = int(x)
        # quotient and remainder for Rice coding
        q = x >> K if K > 0 else x
        r = x & mask if K > 0 else 0

        # write unary(q) then write K-bit remainder
        bw.write_unary_ones_then_zero(q)
        if K > 0:
            bw.write_bits(r, K)

    payload = bw.to_bytes_and_pad()  # finish + pad last byte
    return payload, bw.bit_count


def rice_decode_unsigned(payload, total_bits, K, n_values):
    K = int(K)
    br = BitReader(payload, total_bits)  # read bits from this
    out = np.zeros(int(n_values), dtype=np.uint64)

    for i in range(int(n_values)):
        # read unary quotient and then K remainder bits
        q = br.read_unary_q()
        r = br.read_bits(K) if K > 0 else 0

        # rebuild the original value
        out[i] = (q << K) + r if K > 0 else q

    return out


# # Fixed Predictors

# In[5]:


def fixed_pred_encode(samples, nchannels, order):
    """
    Make residuals using a fixed predictor (order 0..4).
    First 'order' samples are kept as-is (needed to start prediction).
    """
    order = int(order)
    if order < 0 or order > 4:
        raise ValueError("order must be 0..4")

    # ----- mono case -----
    if nchannels == 1:
        x = samples
        N = len(x)
        e = np.empty_like(x)  # residuals

        if order == 0:
            # no prediction, residual = original
            e[:] = x
            return e

        # store first few samples directly (warm-up)
        e[:min(order, N)] = x[:min(order, N)]

        for n in range(order, N):
            # predict using previous samples
            if order == 1:
                pred = x[n-1]
            elif order == 2:
                pred = 2*x[n-1] - x[n-2]
            elif order == 3:
                pred = 3*x[n-1] - 3*x[n-2] + x[n-3]
            else:  # order == 4
                pred = 4*x[n-1] - 6*x[n-2] + 4*x[n-3] - x[n-4]

            # residual = actual - predicted
            e[n] = x[n] - pred

        return e

    # ----- multi-channel case (reshape into frames x channels) -----
    X = samples.reshape(-1, nchannels)
    T = X.shape[0]
    E = np.empty_like(X)

    if order == 0:
        # no prediction, residual = original
        E[:, :] = X[:, :]
        return E.reshape(-1)

    # warm-up frames
    E[:min(order, T), :] = X[:min(order, T), :]

    for n in range(order, T):
        if order == 1:
            pred = X[n-1, :]
        elif order == 2:
            pred = 2*X[n-1, :] - X[n-2, :]
        elif order == 3:
            pred = 3*X[n-1, :] - 3*X[n-2, :] + X[n-3, :]
        else:  # order == 4
            pred = 4*X[n-1, :] - 6*X[n-2, :] + 4*X[n-3, :] - X[n-4, :]

        # residuals per channel
        E[n, :] = X[n, :] - pred

    return E.reshape(-1)


def fixed_pred_decode(residuals, nchannels, order):
    # rebuild original samples from residuals
    order = int(order)
    if order < 0 or order > 4:
        raise ValueError("order must be 0..4")

    # ----- mono case -----
    if nchannels == 1:
        e = residuals
        N = len(e)
        x = np.empty_like(e)

        if order == 0:
            # no prediction used, just copy back
            x[:] = e
            return x

        # warm-up samples (stored raw during encoding)
        x[:min(order, N)] = e[:min(order, N)]

        for n in range(order, N):
            # compute the same predictor (but using already-decoded x)
            if order == 1:
                pred = x[n-1]
            elif order == 2:
                pred = 2*x[n-1] - x[n-2]
            elif order == 3:
                pred = 3*x[n-1] - 3*x[n-2] + x[n-3]
            else:  # order == 4
                pred = 4*x[n-1] - 6*x[n-2] + 4*x[n-3] - x[n-4]

            # original = residual + predicted
            x[n] = e[n] + pred

        return x

    # ----- multi-channel case -----
    E = residuals.reshape(-1, nchannels)
    T = E.shape[0]
    X = np.empty_like(E)

    if order == 0:
        X[:, :] = E[:, :]
        return X.reshape(-1)

    # warm-up frames
    X[:min(order, T), :] = E[:min(order, T), :]

    for n in range(order, T):
        if order == 1:
            pred = X[n-1, :]
        elif order == 2:
            pred = 2*X[n-1, :] - X[n-2, :]
        elif order == 3:
            pred = 3*X[n-1, :] - 3*X[n-2, :] + X[n-3, :]
        else:  # order == 4
            pred = 4*X[n-1, :] - 6*X[n-2, :] + 4*X[n-3, :] - X[n-4, :]

        # add residual back
        X[n, :] = E[n, :] + pred

    return X.reshape(-1)


# # Choose Best Predictor

# In[6]:


def estimate_rice_bits_for_signed(residuals_signed, K):
    """
    Compute exact number of bits needed for Rice coding.
    bits per sample = (q + 1) + K
    """
    K = int(K)

    # convert signed residuals to unsigned (zigzag)
    u = signed_to_zigzag(residuals_signed)

    # q is the unary part (quotient)
    q = (u >> K) if K > 0 else u

    # total bits = sum of q + (1+K) per value
    return int(np.sum(q)) + int(u.size) * (1 + K)


def choose_best_predictor_order(samples_s, nchannels, K, candidate_orders=(0,1,2,3,4)):
    # try different predictor orders and pick the best one
    best_order = None
    best_bits = None

    for order in candidate_orders:
        # compute residuals for this order
        res = fixed_pred_encode(samples_s, nchannels, order)

        # estimate how many Rice bits it would use
        bits = estimate_rice_bits_for_signed(res, K)

        # keep the order that gives minimum bits
        if best_bits is None or bits < best_bits:
            best_bits = bits
            best_order = order

    return best_order, best_bits


# # WAV File Handling

# In[7]:


# ============================================================
# WAV IO (reading and writing PCM WAV files)
# ============================================================

@dataclass
class WavInfo:
    # simple structure to store WAV header info
    nchannels: int
    sampwidth: int
    framerate: int
    nframes: int
    comptype: str
    compname: str


def read_wav_pcm(path):
    # read full file (used later for lossless check)
    with open(path, "rb") as f:
        raw = f.read()

    # use wave module to extract header + frames
    with wave.open(path, "rb") as wf:
        info = WavInfo(
            nchannels=wf.getnchannels(),
            sampwidth=wf.getsampwidth(),
            framerate=wf.getframerate(),
            nframes=wf.getnframes(),
            comptype=wf.getcomptype(),
            compname=wf.getcompname(),
        )
        frames = wf.readframes(info.nframes)

    # make sure file is uncompressed PCM
    if info.comptype != "NONE":
        raise ValueError("Expected PCM WAV, got %s" % info.comptype)

    # convert raw bytes to numpy array depending on sample width
    if info.sampwidth == 1:
        # 8-bit is unsigned → convert to signed
        arr = np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128

    elif info.sampwidth == 2:
        arr = np.frombuffer(frames, dtype=np.int16)

    elif info.sampwidth == 3:
        # 24-bit needs manual reconstruction
        b = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        x = (b[:, 0].astype(np.int32) |
             (b[:, 1].astype(np.int32) << 8) |
             (b[:, 2].astype(np.int32) << 16))
        sign = (x & 0x800000) != 0
        x = x | (sign.astype(np.int32) * ~0xFFFFFF)
        arr = x.astype(np.int32)

    elif info.sampwidth == 4:
        arr = np.frombuffer(frames, dtype=np.int32)

    else:
        raise ValueError("Unsupported sample width: %d" % info.sampwidth)

    # return header info + samples (as int64) + raw bytes
    return info, arr.astype(np.int64), raw


def write_wav_pcm(path, info, samples):
    # convert samples back to correct format
    samples = samples.astype(np.int64)

    if info.sampwidth == 1:
        # convert back to unsigned 8-bit
        x = np.clip(samples + 128, 0, 255).astype(np.uint8)
        frames = x.tobytes()

    elif info.sampwidth == 2:
        frames = samples.astype(np.int16).tobytes()

    elif info.sampwidth == 3:
        # split 24-bit into 3 bytes
        x = samples.astype(np.int32)
        b0 = (x & 0xFF).astype(np.uint8)
        b1 = ((x >> 8) & 0xFF).astype(np.uint8)
        b2 = ((x >> 16) & 0xFF).astype(np.uint8)
        frames = np.column_stack([b0, b1, b2]).ravel().tobytes()

    elif info.sampwidth == 4:
        frames = samples.astype(np.int32).tobytes()

    else:
        raise ValueError("Unsupported sample width: %d" % info.sampwidth)

    # write WAV file using original header info
    with wave.open(path, "wb") as wf:
        wf.setnchannels(info.nchannels)
        wf.setsampwidth(info.sampwidth)
        wf.setframerate(info.framerate)
        wf.writeframes(frames)


# # EX2 File Encode/Decode

# In[8]:


# header format for .ex2 file (custom format)
HEADER_FMT_V3 = "<4sBBBBHHIIII"
MAGIC = b"EX2R"   # file identifier
VERSION = 3       # format version


def encode_wav_to_ex2(wav_path, ex2_path, K, pred_order):
    # read original wav file
    info, samples_s, _raw = read_wav_pcm(wav_path)
    nsamples = samples_s.size

    # apply fixed predictor to get residuals
    residuals = fixed_pred_encode(samples_s, info.nchannels, pred_order)

    # convert residuals to unsigned (zigzag)
    values_u = signed_to_zigzag(residuals)

    # Rice encode
    payload, total_bits = rice_encode_unsigned(values_u, K)

    # build header (store all needed info for decoding)
    header = struct.pack(
        HEADER_FMT_V3,
        MAGIC,
        VERSION,
        int(K),
        int(pred_order),
        0,  # reserved
        int(info.nchannels),
        int(info.sampwidth),
        int(info.framerate),
        int(info.nframes),
        int(nsamples),
        int(total_bits),
    )

    # create output folder if needed
    d = os.path.dirname(ex2_path)
    if d:
        os.makedirs(d, exist_ok=True)

    # write header + compressed payload
    with open(ex2_path, "wb") as f:
        f.write(header)
        f.write(payload)


def decode_ex2_to_wav(ex2_path, wav_out_path):
    # read header and payload
    with open(ex2_path, "rb") as f:
        header_bytes = f.read(struct.calcsize(HEADER_FMT_V3))
        payload = f.read()

    # unpack header values
    (magic, version, K, pred_order, _,
     nch, sw, fr, nframes, nsamples, total_bits) = struct.unpack(HEADER_FMT_V3, header_bytes)

    # basic validation
    if magic != MAGIC:
        raise ValueError("Bad .ex2 magic")
    if version != VERSION:
        raise ValueError("Unsupported .ex2 version: %d" % version)

    # Rice decode
    values_u = rice_decode_unsigned(payload, total_bits, K, nsamples)

    # convert back to signed residuals
    residuals = zigzag_to_signed(values_u)

    # reconstruct original samples
    samples_s = fixed_pred_decode(residuals, nch, pred_order)

    # rebuild WAV header info
    info = WavInfo(
        nchannels=nch,
        sampwidth=sw,
        framerate=fr,
        nframes=nframes,
        comptype="NONE",
        compname="not compressed"
    )

    # create output folder if needed
    d = os.path.dirname(wav_out_path)
    if d:
        os.makedirs(d, exist_ok=True)

    # write decoded wav file
    write_wav_pcm(wav_out_path, info, samples_s)


def files_are_identical(path1, path2):
    # simple lossless check (compare raw bytes)
    with open(path1, "rb") as f1, open(path2, "rb") as f2:
        return f1.read() == f2.read()


# # Main Execution

# In[9]:


def run_exercise2(folder="Exercise2_Files",
                  filenames=("Sound1.wav", "Sound2.wav"),
                  Ks=(4, 2),
                  out_dir="exercise2_output",
                  force_rebuild=True):

    # create output folder
    os.makedirs(out_dir, exist_ok=True)
    results = []
    all_deltas = {}

    # filter only valid input wav files
    safe_inputs = []
    for fn in filenames:
        low = fn.lower()
        if not low.endswith(".wav"):
            continue
        if "_enc" in low or "_dec" in low:
            continue
        safe_inputs.append(fn)

    print("=" * 24)
    print("EXERCISE 2: Rice Coding")
    print("=" * 24)

    for fn in safe_inputs:
        wav_path = os.path.join(folder, fn)
        if not os.path.exists(wav_path):
            raise FileNotFoundError("Missing file: %s" % wav_path)

        original_size = os.path.getsize(wav_path)

        # store results for this file
        row = {"file": fn, "original_size": original_size}

        # read wav only once
        info, samples_s, _raw = read_wav_pcm(wav_path)

        print("\nProcessing:", fn)
        print("-" * 50)

        for K in Ks:
            # choose best predictor order for this K
            best_order, best_bits = choose_best_predictor_order(
                samples_s, info.nchannels, K
            )

            print(f"  Best predictor for K={K}: order={best_order} (est bits={best_bits})")

            # store deltas for plotting later
            deltas = fixed_pred_encode(samples_s, info.nchannels, best_order)
            all_deltas[(fn, K)] = deltas
            
            base = os.path.splitext(fn)[0]
            ex2_path = os.path.join(out_dir, f"{base}_K{K}_Enc.ex2")
            dec_path = os.path.join(out_dir, f"{base}_K{K}_Enc_Dec.wav")

            # encode if needed
            if force_rebuild or (not os.path.exists(ex2_path)):
                print(f"  Encoding K={K}, order={best_order}...", end=" ")
                t0 = time.time()
                encode_wav_to_ex2(wav_path, ex2_path, K, pred_order=best_order)
                print(f"Done ({time.time()-t0:.2f}s)")
            else:
                print("  Skip encode:", os.path.basename(ex2_path))

            # decode if needed
            if force_rebuild or (not os.path.exists(dec_path)):
                print(f"  Decoding K={K}...", end=" ")
                t0 = time.time()
                decode_ex2_to_wav(ex2_path, dec_path)
                print(f"Done ({time.time()-t0:.2f}s)")
            else:
                print("  Skip decode:", os.path.basename(dec_path))

            # check if decoded wav is identical to original
            identical = (
                open(wav_path, "rb").read() ==
                open(dec_path, "rb").read()
            )

            ex2_size = os.path.getsize(ex2_path)
            comp_pct = (1.0 - (float(ex2_size) / float(original_size))) * 100.0

            # store stats
            row[f"rice_k{K}_size"] = ex2_size
            row[f"rice_k{K}_compression_pct"] = comp_pct
            row[f"rice_k{K}_lossless_ok"] = identical
            row[f"pred_order_k{K}"] = best_order

            print(f"    Size: {ex2_size:,} bytes ({comp_pct:+.2f}%)  Lossless={identical}")

        results.append(row)

    return results, all_deltas


# In[10]:


if __name__ == "__main__":
    # run exercise
    results, all_deltas = run_exercise2(
        folder="Exercise2_Files",
        filenames=("Sound1.wav", "Sound2.wav"),
        Ks=(4, 2),
        out_dir="exercise2_output",
        force_rebuild=True
    )


# # Result Table

# In[11]:



    print("\n" + "=" * 14)
    print("RESULTS TABLE")
    print("=" * 14)
    
    df = pd.DataFrame(results)

    # Keep only required columns (no extra rows/columns)
    df = df[[
        "file",
        "original_size",
        "rice_k4_size",
        "rice_k2_size",
        "rice_k4_compression_pct",
        "rice_k2_compression_pct"
    ]]

    # Rename columns to match your image exactly
    df.columns = [
        "File",
        "Original size (bytes)",
        "Rice (K = 4 bits)",
        "Rice (K = 2 bits)",
        "% Compression (K = 4 bits)",
        "% Compression (K = 2 bits)"
    ]

    # Format numbers
    df["Original size (bytes)"] = df["Original size (bytes)"].map("{:,}".format)
    df["Rice (K = 4 bits)"] = df["Rice (K = 4 bits)"].map("{:,}".format)
    df["Rice (K = 2 bits)"] = df["Rice (K = 2 bits)"].map("{:,}".format)
    df["% Compression (K = 4 bits)"] = df["% Compression (K = 4 bits)"].map("{:.2f}".format)
    df["% Compression (K = 2 bits)"] = df["% Compression (K = 2 bits)"].map("{:.2f}".format)


    display(df)


# # Histogram of deltas

# In[12]:


# get deltas for Sound1 with K=4
deltas = all_deltas[("Sound1.wav", 4)]
# plot histogram
plt.figure(figsize=(10, 4))
plt.hist(deltas, bins=150)
plt.title("Histogram of Deltas (Sound1.wav, K=4)")
plt.xlabel("Delta value")
plt.ylabel("Frequency")
plt.show()


# In[13]:


# get deltas for Sound1 with K=2
deltas = all_deltas[("Sound1.wav", 2)]
# plot histogram
plt.figure(figsize=(10, 4))
plt.hist(deltas, bins=150)
plt.title("Histogram of Deltas (Sound1.wav, K=2)")
plt.xlabel("Delta value")
plt.ylabel("Frequency")
plt.show()


# In[14]:


# get deltas for Sound2 with K=4
deltas = all_deltas[("Sound2.wav", 4)]
# plot histogram
plt.figure(figsize=(10, 4))
plt.hist(deltas, bins=150)
plt.title("Histogram of Deltas (Sound2.wav, K=4)")
plt.xlabel("Delta value")
plt.ylabel("Frequency")
plt.show()


# In[15]:


# get deltas for Sound2 with K=2
deltas = all_deltas[("Sound2.wav", 2)]
# plot histogram
plt.figure(figsize=(10, 4))
plt.hist(deltas, bins=150)
plt.title("Histogram of Deltas (Sound2.wav, K=2)")
plt.xlabel("Delta value")
plt.ylabel("Frequency")
plt.show()


# In[ ]:




