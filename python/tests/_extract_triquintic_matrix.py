"""One-shot: parse TRIQUINTIC_COEFFICIENTS[216][216] from the .cuh header
into a numpy .npy file for use by JAX probes."""
import re
import numpy as np

SRC = ("/home/jtufts/src/p312/openmmgridforce/platforms/cuda/src/"
       "kernels/include/TriquinticCoefficients.cuh")

text = open(SRC).read()
m = re.search(r"TRIQUINTIC_COEFFICIENTS\[216\]\[216\]\s*=\s*\{(.*?)\};",
              text, re.DOTALL)
assert m, "Failed to locate coefficient matrix"
body = m.group(1)

# Extract every integer (positive or negative) from the body.
ints = [int(x) for x in re.findall(r"-?\d+", body)]
assert len(ints) == 216 * 216, f"got {len(ints)} ints, expected {216*216}"

M = np.array(ints, dtype=np.int32).reshape(216, 216)
np.save("/home/jtufts/src/p312/openmmgridforce/python/tests/triquintic_matrix.npy", M)
print(f"saved: shape={M.shape}  dtype={M.dtype}  "
      f"nnz={int(np.count_nonzero(M))}/{216*216}  "
      f"range=[{M.min()},{M.max()}]")
