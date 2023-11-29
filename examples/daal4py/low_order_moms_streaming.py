# ==============================================================================
# Copyright 2014 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# daal4py low order moments example for streaming on shared memory systems


from pathlib import Path

from readcsv import pd_read_csv

import daal4py as d4p


def main(readcsv=pd_read_csv, *args, **kwargs):
    # read data from file
    data_path = Path(__file__).parent / "data" / "batch"
    file = data_path / "covcormoments_dense.csv"

    # Configure a low order moments object for streaming
    algo = d4p.low_order_moments(streaming=True)

    chunk_size = 55
    lines_read = 0
    # read and feed chunk by chunk
    while True:
        # Read data in chunks
        try:
            data = readcsv(
                file, usecols=range(10), skip_header=lines_read, max_rows=chunk_size
            )
        except StopIteration as e:
            if lines_read > 0:
                break
            else:
                raise ValueError("No training data was read - empty input file?") from e
        # Now feed chunk
        algo.compute(data)
        lines_read += data.shape[0]

    # All files are done, now finalize the computation
    result = algo.finalize()

    # result provides minimum, maximum, sum, sumSquares, sumSquaresCentered,
    # mean, secondOrderRawMoment, variance, standardDeviation, variation
    return result


if __name__ == "__main__":
    res = main()
    # print results
    print("\nMinimum:\n", res.minimum)
    print("\nMaximum:\n", res.maximum)
    print("\nSum:\n", res.sum)
    print("\nSum of squares:\n", res.sumSquares)
    print("\nSum of squared difference from the means:\n", res.sumSquaresCentered)
    print("\nMean:\n", res.mean)
    print("\nSecond order raw moment:\n", res.secondOrderRawMoment)
    print("\nVariance:\n", res.variance)
    print("\nStandard deviation:\n", res.standardDeviation)
    print("\nVariation:\n", res.variation)
    print("All looks good!")
