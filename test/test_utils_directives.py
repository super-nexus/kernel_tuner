from pytest import raises

from kernel_tuner.utils.directives import *


def test_correct_kernel():
    assert correct_kernel("vector_add", "tuner start vector_add")
    assert correct_kernel("vector_add", "tuner start vector_add a(float:size)")
    assert not correct_kernel("vector_add", "tuner start gemm")
    assert not correct_kernel("vector_add", "tuner start gemm a(float:size) b(float:size)")


def test_is_cpp():
    cpp_code = "int main(void) {\n#pragma acc parallel}"
    assert is_cpp(cpp_code, "openacc")
    assert is_cpp(cpp_code, "OpenACC")
    assert not is_cpp(cpp_code, "open acc")


def test_is_f90():
    f90_code = "!$acc parallel"
    assert is_f90(f90_code, "openacc")
    assert is_f90(f90_code, "OpenACC")
    assert not is_f90(f90_code, "open acc")


def test_is_cpp_or_f90():
    cpp_code = "int main(void) {\n#pragma acc parallel}"
    f90_code = "!$acc parallel"
    one, two = is_cpp_or_f90(cpp_code)
    assert one
    assert not two
    one, two = is_cpp_or_f90(f90_code)
    assert not one
    assert two


def test_parse_size():
    assert parse_size(128) == 128
    assert parse_size("16") == 16
    assert parse_size("test") is None
    assert parse_size("n", ["#define n 1024\n"]) == 1024
    assert parse_size("n,m", ["#define n 16\n", "#define m 32\n"]) == 512
    assert parse_size("n", ["#define size 512\n"], {"n": 32}) == 32
    assert parse_size("m", ["#define size 512\n"], {"n": 32}) is None
    assert parse_size("rows,cols", dimensions={"rows": 16, "cols": 8}) == 128


def test_create_data_directive():
    assert (
        create_data_directive("array", 1024, True, False)
        == "#pragma acc enter data create(array[1024])\n#pragma acc update device(array[1024])\n"
    )
    assert (
        create_data_directive("matrix", 35, False, True)
        == "!$acc enter data create(matrix(35))\n!$acc update device(matrix(35))\n"
    )


def test_exit_data_directive():
    assert exit_data_directive("array", 1024, True, False) == "#pragma acc exit data copyout(array[1024])\n"
    assert exit_data_directive("matrix", 35, False, True) == "!$acc exit data copyout(matrix(35))\n"


def test_extract_directive_code():
    code = """
        #include <stdlib.h>

        #define VECTOR_SIZE 65536

        int main(void) {
            int size = VECTOR_SIZE;
            __restrict float * a = (float *) malloc(VECTOR_SIZE * sizeof(float));
            __restrict float * b = (float *) malloc(VECTOR_SIZE * sizeof(float));
            __restrict float * c = (float *) malloc(VECTOR_SIZE * sizeof(float));

            #pragma tuner start initialize
            #pragma acc parallel
            #pragma acc loop
            for ( int i = 0; i < size; i++ ) {
                    a[i] = i;
                    b[i] = i + 1;
            }
            #pragma tuner stop

            #pragma tuner start vector_add
            #pragma acc parallel
            #pragma acc loop
            for ( int i = 0; i < size; i++ ) {
                    c[i] = a[i] + b[i];
            }
            #pragma tuner stop

            free(a);
            free(b);
            free(c);
    }
    """
    expected_one = """            #pragma acc parallel
            #pragma acc loop
            for ( int i = 0; i < size; i++ ) {
                    a[i] = i;
                    b[i] = i + 1;
            }"""
    expected_two = """            #pragma acc parallel
            #pragma acc loop
            for ( int i = 0; i < size; i++ ) {
                    c[i] = a[i] + b[i];
            }"""
    returns = extract_directive_code(code)
    assert len(returns) == 2
    assert expected_one in returns["initialize"]
    assert expected_two in returns["vector_add"]
    assert expected_one not in returns["vector_add"]
    returns = extract_directive_code(code, "vector")
    assert len(returns) == 0

    code = """
    !$tuner start vector_add
    !$acc parallel loop num_gangs(ngangs) vector_length(vlength)
    do i = 1, N
      C(i) = A(i) + B(i)
    end do
    !$acc end parallel loop
    !$tuner stop
    """
    expected = """    !$acc parallel loop num_gangs(ngangs) vector_length(vlength)
    do i = 1, N
      C(i) = A(i) + B(i)
    end do
    !$acc end parallel loop"""
    returns = extract_directive_code(code, "vector_add")
    assert len(returns) == 1
    assert expected in returns["vector_add"]


def test_extract_preprocessor():
    code = """
        #include <stdlib.h>

        #define VECTOR_SIZE 65536

        int main(void) {
            int size = VECTOR_SIZE;
            __restrict float * a = (float *) malloc(VECTOR_SIZE * sizeof(float));
            __restrict float * b = (float *) malloc(VECTOR_SIZE * sizeof(float));
            __restrict float * c = (float *) malloc(VECTOR_SIZE * sizeof(float));

            #pragma tuner start
            #pragma acc parallel
            #pragma acc loop
            for ( int i = 0; i < size; i++ ) {
                    a[i] = i;
                    b[i] = i + 1;
            }
            #pragma tuner stop

            #pragma tuner start
            #pragma acc parallel
            #pragma acc loop
            for ( int i = 0; i < size; i++ ) {
                    c[i] = a[i] + b[i];
            }
            #pragma tuner stop

            free(a);
            free(b);
            free(c);
    }
    """
    expected = ["        #include <stdlib.h>", "        #define VECTOR_SIZE 65536"]
    results = extract_preprocessor(code)
    assert len(results) == 2
    for item in expected:
        assert item in results


def test_wrap_timing():
    code = "#pragma acc\nfor ( int i = 0; i < size; i++ ) {\nc[i] = a[i] + b[i];\n}"
    wrapped = wrap_timing(code)
    wrapped = close_cpp_timing(wrapped)
    assert (
        wrapped
        == "auto kt_timing_start = std::chrono::steady_clock::now();\n#pragma acc\nfor ( int i = 0; i < size; i++ ) {\nc[i] = a[i] + b[i];\n}\nauto kt_timing_end = std::chrono::steady_clock::now();\nstd::chrono::duration<float, std::milli> elapsed_time = kt_timing_end - kt_timing_start;\nreturn elapsed_time.count();\n"
    )


def test_wrap_data():
    code_cpp = "// this is a comment\n"
    code_f90 = "! this is a comment\n"
    data = {"array": ["int*", "size"]}
    preprocessor = ["#define size 42"]
    expected_cpp = "#pragma acc enter data create(array[42])\n#pragma acc update device(array[42])\n// this is a comment\n#pragma acc exit data copyout(array[42])\n"
    assert wrap_data(code_cpp, data, preprocessor, None, True, False) == expected_cpp
    expected_f90 = "!$acc enter data create(array(42))\n!$acc update device(array(42))\n! this is a comment\n!$acc exit data copyout(array(42))\n"
    assert wrap_data(code_f90, data, preprocessor, None, False, True) == expected_f90


def test_extract_directive_signature():
    code = "#pragma tuner start vector_add a(float*:VECTOR_SIZE) b(float*:VECTOR_SIZE) c(float*:VECTOR_SIZE) size(int:VECTOR_SIZE)  \n#pragma acc"
    signatures = extract_directive_signature(code)
    assert len(signatures) == 1
    assert (
        "float vector_add(float * restrict a, float * restrict b, float * restrict c, int size)"
        in signatures["vector_add"]
    )
    signatures = extract_directive_signature(code, "vector_add")
    assert len(signatures) == 1
    assert (
        "float vector_add(float * restrict a, float * restrict b, float * restrict c, int size)"
        in signatures["vector_add"]
    )
    signatures = extract_directive_signature(code, "vector_add_ext")
    assert len(signatures) == 0
    code = "!$tuner start vector_add A(float*:VECTOR_SIZE) B(float*:VECTOR_SIZE) C(float*:VECTOR_SIZE) n(int:VECTOR_SIZE)\n!$acc"
    signatures = extract_directive_signature(code)
    assert len(signatures) == 1
    assert "function vector_add(A, B, C, n)" in signatures["vector_add"]


def test_extract_directive_data():
    code = "#pragma tuner start vector_add a(float*:VECTOR_SIZE) b(float*:VECTOR_SIZE) c(float*:VECTOR_SIZE) size(int:VECTOR_SIZE)\n#pragma acc"
    data = extract_directive_data(code)
    assert len(data) == 1
    assert len(data["vector_add"]) == 4
    assert "float*" in data["vector_add"]["b"]
    assert "int" not in data["vector_add"]["c"]
    assert "VECTOR_SIZE" in data["vector_add"]["size"]
    data = extract_directive_data(code, "vector_add_double")
    assert len(data) == 0
    code = "!$tuner start vector_add A(float*:VECTOR_SIZE) B(float*:VECTOR_SIZE) C(float*:VECTOR_SIZE) n(int:VECTOR_SIZE)\n!$acc"
    data = extract_directive_data(code)
    assert len(data) == 1
    assert len(data["vector_add"]) == 4
    assert "float*" in data["vector_add"]["B"]
    assert "int" not in data["vector_add"]["C"]
    assert "VECTOR_SIZE" in data["vector_add"]["n"]
    code = (
        "!$tuner start matrix_add A(float*:N_ROWS,N_COLS) B(float*:N_ROWS,N_COLS) nr(int:N_ROWS) nc(int:N_COLS)\n!$acc"
    )
    data = extract_directive_data(code)
    assert len(data) == 1
    assert len(data["matrix_add"]) == 4
    assert "float*" in data["matrix_add"]["A"]
    assert "N_ROWS,N_COLS" in data["matrix_add"]["B"]


def test_allocate_signature_memory():
    code = "#pragma tuner start vector_add a(float*:VECTOR_SIZE) b(float*:VECTOR_SIZE) c(float*:VECTOR_SIZE) size(int:VECTOR_SIZE)\n#pragma acc"
    data = extract_directive_data(code)
    with raises(TypeError):
        _ = allocate_signature_memory(data["vector_add"])
    preprocessor = ["#define VECTOR_SIZE 1024\n"]
    args = allocate_signature_memory(data["vector_add"], preprocessor)
    assert type(args[0]) is np.ndarray
    assert type(args[1]) is not np.float64
    assert args[2].dtype == "float32"
    assert type(args[3]) is np.int32
    assert args[3] == 1024
    user_values = dict()
    user_values["VECTOR_SIZE"] = 1024
    args = allocate_signature_memory(data["vector_add"], user_dimensions=user_values)
    assert type(args[0]) is np.ndarray
    assert type(args[1]) is not np.float64
    assert args[2].dtype == "float32"
    assert type(args[3]) is np.int32
    code = (
        "!$tuner start matrix_add A(float*:N_ROWS,N_COLS) B(float*:N_ROWS,N_COLS) nr(int:N_ROWS) nc(int:N_COLS)\n!$acc"
    )
    data = extract_directive_data(code)
    preprocessor = ["#define N_ROWS 128\n", "#define N_COLS 512\n"]
    args = allocate_signature_memory(data["matrix_add"], preprocessor)
    assert args[2] == 128
    assert len(args[0]) == (128 * 512)
    user_values = dict()
    user_values["N_ROWS"] = 32
    user_values["N_COLS"] = 16
    args = allocate_signature_memory(data["matrix_add"], user_dimensions=user_values)
    assert args[3] == 16
    assert len(args[1]) == 512
