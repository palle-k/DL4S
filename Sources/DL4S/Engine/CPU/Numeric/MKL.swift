//
//  MKL.swift
//  DL4S
//
//  Created by Palle Klewitz on 20.10.19.
//  Copyright (c) 2019 - Palle Klewitz
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

// MKL to Accelerate Bridge
#if !canImport(Accelerate) && canImport(MKL)
import MKL
typealias CBLAS_ORDER = CBLAS_LAYOUT
typealias CBLAS_TRANSPOSE = MKL.CBLAS_TRANSPOSE

let CblasRowMajor = CBLAS_ORDER(rawValue: 101)
let CblasColMajor = CBLAS_ORDER(rawValue: 102)

let CblasNoTrans = CBLAS_TRANSPOSE(rawValue: 111)
let CblasTrans = CBLAS_TRANSPOSE(rawValue: 112)


typealias vDSP_Stride = Int
typealias vDSP_Length = UInt

// MARK: vDSP Float32
func vDSP_vfill(_ __A: UnsafePointer<Float>, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[0]
    }
}

func __callMkl(
    lhs: UnsafePointer<Float>,
    rhs: UnsafePointer<Float>, 
    result: UnsafeMutablePointer<Float>, 
    strideLhs: vDSP_Stride,
    strideRhs: vDSP_Stride, 
    strideResult: vDSP_Stride, 
    count: vDSP_Length, 
    operation: (UnsafePointer<Float>, UnsafePointer<Float>, UnsafeMutablePointer<Float>) -> ()
) {
    let lhsInput: UnsafePointer<Float>
    var lhsPtr: UnsafeMutablePointer<Float>? = nil
    if strideLhs == 1 {
        lhsInput = lhs
    } else {
        let lhsBuffer = UnsafeMutablePointer<Float>.allocate(capacity: Int(count))
        lhsPtr = lhsBuffer
        vsPackI(Int32(count), lhs, Int32(strideLhs), lhsBuffer)
        lhsInput = UnsafePointer(lhsBuffer)
    }
    let rhsInput: UnsafePointer<Float>
    var rhsPtr: UnsafeMutablePointer<Float>? = nil
    if strideRhs == 1 {
        rhsInput = rhs
    } else {
        let rhsBuffer = UnsafeMutablePointer<Float>.allocate(capacity: Int(count))
        rhsPtr = rhsBuffer
        vsPackI(Int32(count), rhs, Int32(strideRhs), rhsBuffer)
        rhsInput = UnsafePointer(rhsBuffer)
    }
    let resultInput: UnsafeMutablePointer<Float>
    if strideResult == 1 {
        resultInput = result
    } else {
        resultInput = UnsafeMutablePointer<Float>.allocate(capacity: Int(count))
    }
    operation(lhsInput, rhsInput, resultInput)
    lhsPtr?.deallocate()
    rhsPtr?.deallocate()

    if strideResult != 1 {
        vsUnpackI(Int32(count), resultInput, result, Int32(strideResult))
        resultInput.deallocate()
    }
}

func __callMkl(
    input: UnsafePointer<Float>,
    result: UnsafeMutablePointer<Float>, 
    strideInput: vDSP_Stride,
    strideResult: vDSP_Stride, 
    count: vDSP_Length, 
    operation: (UnsafePointer<Float>, UnsafeMutablePointer<Float>) -> ()
) {
    let inputVals: UnsafePointer<Float>
    var inputPtr: UnsafeMutablePointer<Float>? = nil
    if strideInput == 1 {
        inputVals = input
    } else {
        let inputBuffer = UnsafeMutablePointer<Float>.allocate(capacity: Int(count))
        inputPtr = inputBuffer
        vsPackI(Int32(count), input, Int32(strideInput), inputBuffer)
        inputVals = UnsafePointer(inputBuffer)
    }
    let resultInput: UnsafeMutablePointer<Float>
    if strideResult == 1 {
        resultInput = result
    } else {
        resultInput = UnsafeMutablePointer<Float>.allocate(capacity: Int(count))
    }
    operation(inputVals, resultInput)
    inputPtr?.deallocate()

    if strideResult != 1 {
        vsUnpackI(Int32(count), resultInput, result, Int32(strideResult))
        resultInput.deallocate()
    }
}
func vDSP_vsq(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    if __IA == 1 && __IC == 1 {
        vsSqr(Int32(__N), __A, __C)
    } else if __IA == 1 {
        let tmp = UnsafeMutablePointer<Float>.allocate(capacity: Int(__N))
        defer {
            tmp.deallocate()
        }
        vsSqr(Int32(__N), __A, tmp)
        vsUnpackI(Int32(__N), tmp, __C, Int32(__IC))
    } else if __IC == 1 {
        let tmp = UnsafeMutablePointer<Float>.allocate(capacity: Int(__N))
        defer {
            tmp.deallocate()
        }
        vsPackI(Int32(__N), __A, Int32(__IA), tmp)
        vsSqr(Int32(__N), tmp, __C)
    } else {
        let tmp = UnsafeMutablePointer<Float>.allocate(capacity: Int(__N))
        defer {
            tmp.deallocate()
        }
        vsPackI(Int32(__N), __A, Int32(__IA), tmp)
        vsSqr(Int32(__N), tmp, tmp)
        vsUnpackI(Int32(__N), tmp, __C, Int32(__IC))
    }
}

func vDSP_vthr(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = max(__A[i * __IA], __B[0])
    }
}

func vDSP_vsadd(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    cblas_scopy(Int32(__N), __A, Int32(__IA), __C, Int32(__IC))
    cblas_saxpy(Int32(__N), 1, __B, 0, __C, Int32(__IC))
}

func vDSP_vsmul(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    cblas_scopy(Int32(__N), __A, Int32(__IA), __C, Int32(__IC))
    cblas_sscal(Int32(__N), __B[0], __C, Int32(__IC))
}

func vDSP_svdiv(_ __A: UnsafePointer<Float>, _ __B: UnsafePointer<Float>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[0] / __B[i * __IB]
    }
}

func vDSP_vadd(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    __callMkl(lhs: __A, rhs: __B, result: __C, strideLhs: __IA, strideRhs: __IB, strideResult: __IC, count: __N) { lhs, rhs, result in
        vsAdd(Int32(__N), lhs, rhs, result)
    }
}

func vDSP_vmul(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    __callMkl(lhs: __A, rhs: __B, result: __C, strideLhs: __IA, strideRhs: __IB, strideResult: __IC, count: __N) { lhs, rhs, result in
        vsMul(Int32(__N), lhs, rhs, result)
    }
}

func vDSP_vneg(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    cblas_scopy(Int32(__N), __A, Int32(__IA), __C, Int32(__IC))
    cblas_sscal(Int32(__N), -1, __C, Int32(__IC))

}

func vDSP_vsub(_ __B: UnsafePointer<Float>, _ __IB: vDSP_Stride, _ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    __callMkl(lhs: __A, rhs: __B, result: __C, strideLhs: __IA, strideRhs: __IB, strideResult: __IC, count: __N) { lhs, rhs, result in
        vsSub(Int32(__N), lhs, rhs, result)
    }
}

func vDSP_vma(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __IB: vDSP_Stride, _ __C: UnsafePointer<Float>, _ __IC: vDSP_Stride, _ __D: UnsafeMutablePointer<Float>, _ __ID: vDSP_Stride, _ __N: vDSP_Length) {
    vDSP_vmul(__A, __IA, __B, __IB, __D, __ID, __N)
    vDSP_vadd(__C, __IC, __D, __ID, __D, __ID, __N)
}

func vDSP_vdiv(_ __B: UnsafePointer<Float>, _ __IB: vDSP_Stride, _ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    __callMkl(lhs: __A, rhs: __B, result: __C, strideLhs: __IA, strideRhs: __IB, strideResult: __IC, count: __N) { lhs, rhs, result in
        vsDiv(Int32(__N), lhs, rhs, result)
    }
}

func vDSP_sve(_ __A: UnsafePointer<Float>, _ __I: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __N: vDSP_Length) {
    __C[0] = cblas_sdot(Int32(__N), __A, Int32(__I), [0], 0)
}

func vDSP_vmsa(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __IB: vDSP_Stride, _ __C: UnsafePointer<Float>, _ __D: UnsafeMutablePointer<Float>, _ __ID: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __D[i * __ID] = __A[i * __IA] * __B[i * __IB] + __C[0]
    }
}

func vDSP_vsma(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __C: UnsafePointer<Float>, _ __IC: vDSP_Stride, _ __D: UnsafeMutablePointer<Float>, _ __ID: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __D[i * __ID] = __A[i * __IA] * __B[0] + __C[i * __IC]
    }
}

func vDSP_mtrans(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __M: vDSP_Length, _ __N: vDSP_Length) {
    let c = Int(__M)
    let r = Int(__N)
    for src_col in 0 ..< c {
        for src_row in 0 ..< r {
            __C[(src_col * r + src_row) * __IC] = __A[(src_col + src_row * c) * __IA]
        }
    }
}

func vDSP_dotpr(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __N: vDSP_Length) {
    __C[0] = cblas_sdot(Int32(__N), __A, Int32(__IA), __B, Int32(__IB))
}

func vDSP_maxvi(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __I: UnsafeMutablePointer<vDSP_Length>, _ __N: vDSP_Length) {
    __I[0] = vDSP_Length(
        cblas_isamax(Int32(__N), __A, Int32(__IA))
    )
    __C[0] = __A[Int(__I[0])]
}

func vDSP_minvi(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __I: UnsafeMutablePointer<vDSP_Length>, _ __N: vDSP_Length) {
    __I[0] = vDSP_Length(
        cblas_isamin(Int32(__N), __A, Int32(__IA))
    )
    __C[0] = __A[Int(__I[0])]
}

func vDSP_vmax(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    __C[0] = __A[cblas_isamax(Int32(__N), __A, Int32(__IA))]
}

func vDSP_vmin(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    __C[0] = __A[cblas_isamin(Int32(__N), __A, Int32(__IA))]
}

func vDSP_vramp(_ __A: UnsafePointer<Float>, _ __B: UnsafePointer<Float>, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[0] + Float(i) * __B[0]
    }
}

// MARK: vDSP Double
func vDSP_vfillD(_ __A: UnsafePointer<Double>, _ __C: UnsafeMutablePointer<Double>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[0]
    }
}

func vDSP_vsqD(_ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Double>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA]
    }
}

func vDSP_vthrD(_ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Double>, _ __C: UnsafeMutablePointer<Double>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = max(__A[i * __IA], __B[0])
    }
}

func vDSP_vsaddD(_ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Double>, _ __C: UnsafeMutablePointer<Double>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA] + __B[0]
    }
}

func vDSP_vsmulD(_ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Double>, _ __C: UnsafeMutablePointer<Double>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA] * __B[0]
    }
}

func vDSP_svdivD(_ __A: UnsafePointer<Double>, _ __B: UnsafePointer<Double>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Double>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[0] / __B[i * __IB]
    }
}

func vDSP_vaddD(_ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Double>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Double>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA] + __B[i * __IB]
    }
}

func vDSP_vmulD(_ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Double>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Double>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA] * __B[i * __IB]
    }
}

func vDSP_vnegD(_ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Double>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = -__A[i * __IA]
    }
}

func vDSP_vsubD(_ __B: UnsafePointer<Double>, _ __IB: vDSP_Stride, _ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Double>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA] - __B[i * __IB]
    }
}

func vDSP_vmaD(_ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Double>, _ __IB: vDSP_Stride, _ __C: UnsafePointer<Double>, _ __IC: vDSP_Stride, _ __D: UnsafeMutablePointer<Double>, _ __ID: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __D[i * __ID] = __A[i * __IA] * __B[i * __IB] + __C[i * __IC]
    }
}

func vDSP_vdivD(_ __B: UnsafePointer<Double>, _ __IB: vDSP_Stride, _ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Double>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA] / __B[i * __IB]
    }
}

func vDSP_sveD(_ __A: UnsafePointer<Double>, _ __I: vDSP_Stride, _ __C: UnsafeMutablePointer<Double>, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[0] += __A[i * __I]
    }
}

func vDSP_vmsaD(_ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Double>, _ __IB: vDSP_Stride, _ __C: UnsafePointer<Double>, _ __D: UnsafeMutablePointer<Double>, _ __ID: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __D[i * __ID] = __A[i * __IA] * __B[i * __IB] + __C[0]
    }
}

func vDSP_vsmaD(_ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Double>, _ __C: UnsafePointer<Double>, _ __IC: vDSP_Stride, _ __D: UnsafeMutablePointer<Double>, _ __ID: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __D[i * __ID] = __A[i * __IA] * __B[0] + __C[i * __IC]
    }
}

func vDSP_mtransD(_ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Double>, _ __IC: vDSP_Stride, _ __M: vDSP_Length, _ __N: vDSP_Length) {
    let c = Int(__M)
    let r = Int(__N)
    for src_col in 0 ..< c {
        for src_row in 0 ..< r {
            __C[(src_col * r + src_row) * __IC] = __A[(src_col + src_row * c) * __IA]
        }
    }
}

func vDSP_dotprD(_ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Double>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Double>, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[0] += __A[i * __IA] + __B[i * __IB]
    }
}

func vDSP_maxviD(_ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Double>, _ __I: UnsafeMutablePointer<vDSP_Length>, _ __N: vDSP_Length) {
    var maxI = -1
    var maxV = -Double.infinity
    
    for i in 0 ..< Int(__N) {
        let v = __A[i * __IA]
        if v > maxV {
            maxV = v
            maxI = i
        }
    }
    
    __I[0] = UInt(maxI)
    __C[0] = maxV
}

func vDSP_minviD(_ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Double>, _ __I: UnsafeMutablePointer<vDSP_Length>, _ __N: vDSP_Length) {
    var minI = -1
    var minV = -Double.infinity
    
    for i in 0 ..< Int(__N) {
        let v = __A[i * __IA]
        if v < minV {
            minV = v
            minI = i
        }
    }
    
    __I[0] = UInt(minI)
    __C[0] = minV
}

func vDSP_vmaxD(_ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Double>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Double>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    var maxV = -Double.infinity
    
    for i in 0 ..< Int(__N) {
        let v = __A[i * __IA]
        if v > maxV {
            maxV = v
        }
    }
    
    __C[0] = maxV
}

func vDSP_vminD(_ __A: UnsafePointer<Double>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Double>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Double>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    var minV = -Double.infinity
    
    for i in 0 ..< Int(__N) {
        let v = __A[i * __IA]
        if v < minV {
            minV = v
        }
    }
    
    __C[0] = minV
}

func vDSP_vrampD(_ __A: UnsafePointer<Double>, _ __B: UnsafePointer<Double>, _ __C: UnsafeMutablePointer<Double>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[0] + Double(i) * __B[0]
    }
}

// MARK: vDSP Int32

func vDSP_vfilli(_ __A: UnsafePointer<Int32>, _ __C: UnsafeMutablePointer<Int32>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[0]
    }
}

func vDSP_vsqi(_ __A: UnsafePointer<Int32>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Int32>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA]
    }
}

func vDSP_vthri(_ __A: UnsafePointer<Int32>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Int32>, _ __C: UnsafeMutablePointer<Int32>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = max(__A[i * __IA], __B[0])
    }
}

func vDSP_vsaddi(_ __A: UnsafePointer<Int32>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Int32>, _ __C: UnsafeMutablePointer<Int32>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA] + __B[0]
    }
}

func vDSP_vsmuli(_ __A: UnsafePointer<Int32>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Int32>, _ __C: UnsafeMutablePointer<Int32>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA] * __B[0]
    }
}

func vDSP_svdivi(_ __A: UnsafePointer<Int32>, _ __B: UnsafePointer<Int32>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Int32>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[0] / __B[i * __IB]
    }
}

func vDSP_vaddi(_ __A: UnsafePointer<Int32>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Int32>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Int32>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA] + __B[i * __IB]
    }
}

func vDSP_vmuli(_ __A: UnsafePointer<Int32>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Int32>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Int32>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA] * __B[i * __IB]
    }
}

func vDSP_vnegi(_ __A: UnsafePointer<Int32>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Int32>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = -__A[i * __IA]
    }
}

func vDSP_vsubi(_ __B: UnsafePointer<Int32>, _ __IB: vDSP_Stride, _ __A: UnsafePointer<Int32>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Int32>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA] - __B[i * __IB]
    }
}

func vDSP_vmai(_ __A: UnsafePointer<Int32>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Int32>, _ __IB: vDSP_Stride, _ __C: UnsafePointer<Int32>, _ __IC: vDSP_Stride, _ __D: UnsafeMutablePointer<Int32>, _ __ID: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __D[i * __ID] = __A[i * __IA] * __B[i * __IB] + __C[i * __IC]
    }
}

func vDSP_vdivi(_ __B: UnsafePointer<Int32>, _ __IB: vDSP_Stride, _ __A: UnsafePointer<Int32>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Int32>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA] / __B[i * __IB]
    }
}

func vDSP_svei(_ __A: UnsafePointer<Int32>, _ __I: vDSP_Stride, _ __C: UnsafeMutablePointer<Int32>, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[0] += __A[i * __I]
    }
}

func vDSP_vmsai(_ __A: UnsafePointer<Int32>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Int32>, _ __IB: vDSP_Stride, _ __C: UnsafePointer<Int32>, _ __D: UnsafeMutablePointer<Int32>, _ __ID: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __D[i * __ID] = __A[i * __IA] * __B[i * __IB] + __C[0]
    }
}

func vDSP_vsmai(_ __A: UnsafePointer<Int32>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Int32>, _ __C: UnsafePointer<Int32>, _ __IC: vDSP_Stride, _ __D: UnsafeMutablePointer<Int32>, _ __ID: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __D[i * __ID] = __A[i * __IA] * __B[0] + __C[i * __IC]
    }
}

func vDSP_mtransi(_ __A: UnsafePointer<Int32>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Int32>, _ __IC: vDSP_Stride, _ __M: vDSP_Length, _ __N: vDSP_Length) {
    let c = Int(__M)
    let r = Int(__N)
    for src_col in 0 ..< c {
        for src_row in 0 ..< r {
            __C[(src_col * r + src_row) * __IC] = __A[(src_col + src_row * c) * __IA]
        }
    }
}

func vDSP_dotpri(_ __A: UnsafePointer<Int32>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Int32>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Int32>, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[0] += __A[i * __IA] + __B[i * __IB]
    }
}

func vDSP_maxvii(_ __A: UnsafePointer<Int32>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Int32>, _ __I: UnsafeMutablePointer<vDSP_Length>, _ __N: vDSP_Length) {
    var maxI = -1
    var maxV = Int32.min
    
    for i in 0 ..< Int(__N) {
        let v = __A[i * __IA]
        if v > maxV {
            maxV = v
            maxI = i
        }
    }
    
    __I[0] = UInt(maxI)
    __C[0] = maxV
}

func vDSP_minvii(_ __A: UnsafePointer<Int32>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Int32>, _ __I: UnsafeMutablePointer<vDSP_Length>, _ __N: vDSP_Length) {
    var minI = -1
    var minV = Int32.min
    
    for i in 0 ..< Int(__N) {
        let v = __A[i * __IA]
        if v < minV {
            minV = v
            minI = i
        }
    }
    
    __I[0] = UInt(minI)
    __C[0] = minV
}

func vDSP_vmaxi(_ __A: UnsafePointer<Int32>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Int32>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Int32>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    var maxV = Int32.min
    
    for i in 0 ..< Int(__N) {
        let v = __A[i * __IA]
        if v > maxV {
            maxV = v
        }
    }
    
    __C[0] = maxV
}

func vDSP_vmini(_ __A: UnsafePointer<Int32>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Int32>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Int32>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    var minV = Int32.min
    
    for i in 0 ..< Int(__N) {
        let v = __A[i * __IA]
        if v < minV {
            minV = v
        }
    }
    
    __C[0] = minV
}

func vDSP_vrampi(_ __A: UnsafePointer<Int32>, _ __B: UnsafePointer<Int32>, _ __C: UnsafeMutablePointer<Int32>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[0] + Int32(i) * __B[0]
    }
}

// MARK: veclib Float

func vvtanhf(_ c: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ l: UnsafePointer<Int32>) {
    vsTanh(l[0], a, c)
}

func vvsqrtf(_ c: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ l: UnsafePointer<Int32>) {
    vsSqrt(l[0], a, c)
}

func vvexpf(_ c: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ l: UnsafePointer<Int32>) {
    vsExp(l[0], a, c)
}

func vvlogf(_ c: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ l: UnsafePointer<Int32>) {
    vsLn(l[0], a, c)
}

func vvsinf(_ c: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ l: UnsafePointer<Int32>) {
    vsSin(l[0], a, c)
}

func vvcosf(_ c: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ l: UnsafePointer<Int32>) {
    vsCos(l[0], a, c)
}

func vvtanf(_ c: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ l: UnsafePointer<Int32>) {
    vsTan(l[0], a, c)
}

func vvcopysignf(_ c: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ s: UnsafePointer<Float>, _ l: UnsafePointer<Int32>) {
    for i in 0 ..< Int(l[0]) {
        c[i] = copysignf(a[i], s[i])
    }
}

// MARK: veclib Double
func vvtanh(_ c: UnsafeMutablePointer<Double>, _ a: UnsafePointer<Double>, _ l: UnsafePointer<Int32>) {
    vdTanh(l[0], a, c)
}

func vvsqrt(_ c: UnsafeMutablePointer<Double>, _ a: UnsafePointer<Double>, _ l: UnsafePointer<Int32>) {
    vdSqrt(l[0], a, c)
}

func vvexp(_ c: UnsafeMutablePointer<Double>, _ a: UnsafePointer<Double>, _ l: UnsafePointer<Int32>) {
    vdExp(l[0], a, c)
}

func vvlog(_ c: UnsafeMutablePointer<Double>, _ a: UnsafePointer<Double>, _ l: UnsafePointer<Int32>) {
    vdLn(l[0], a, c)
}

func vvsin(_ c: UnsafeMutablePointer<Double>, _ a: UnsafePointer<Double>, _ l: UnsafePointer<Int32>) {
    vdSin(l[0], a, c)
}

func vvcos(_ c: UnsafeMutablePointer<Double>, _ a: UnsafePointer<Double>, _ l: UnsafePointer<Int32>) {
    vdCos(l[0], a, c)
}

func vvtan(_ c: UnsafeMutablePointer<Double>, _ a: UnsafePointer<Double>, _ l: UnsafePointer<Int32>) {
    vdTan(l[0], a, c)
}

func vvcopysign(_ c: UnsafeMutablePointer<Double>, _ a: UnsafePointer<Double>, _ s: UnsafePointer<Double>, _ l: UnsafePointer<Int32>) {
    for i in 0 ..< Int(l[0]) {
        c[i] = copysign(a[i], s[i])
    }
}

#endif