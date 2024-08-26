# Test_of_GroupedGemm_on_multiGpus
This is a test to GroupedGemm on a multi-gpu environment which has encountered Cuda error. The related issue on

The **original_cases_on_singleGpu.py** is the original file for testing Grouped-Gemm bases on a single-gpu environment, and it's from the following git repository: https://github.com/tgale96/grouped_gemm/blob/main/grouped_gemm/ops_test.py

The the **badcases_1st.py** is the first bad case on a multi-gpu environment. This test file specifies the device(device ID as 3) when creating a new Tensor compared with the original file.

The the **badcases_2nd.py** runs well on a multi-gpu environment and compared to the 1st test, this file specifies the device ID as 0 when creating the tensor.

The the **badcases_3rd.py** is the actually the 2nd bad case on a multi-gpu environment and compared to the 1st test, this file uses random device ID when creating the tensor.
