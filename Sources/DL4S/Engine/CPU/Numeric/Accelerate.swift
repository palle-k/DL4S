//
//  Accelerate.swift
//  DL4S
//
//  Created by Palle Klewitz on 20.10.19.
//

import Foundation

#if !canImport(Accelerate)

typealias vDSP_Stride = Int
typealias vDSP_Length = UInt

// MARK: vDSP Float32
func vDSP_vfill(_ __A: UnsafePointer<Float>, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[0]
    }
}

func vDSP_vsq(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA]
    }
}

func vDSP_vthr(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = max(__A[i * __IA], __B[0])
    }
}

func vDSP_vsadd(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA] + __B[0]
    }
}

func vDSP_vsmul(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA] * __B[0]
    }
}

func vDSP_svdiv(_ __A: UnsafePointer<Float>, _ __B: UnsafePointer<Float>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[0] / __B[i * __IB]
    }
}

func vDSP_vadd(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA] + __B[i * __IB]
    }
}

func vDSP_vmul(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA] * __B[i * __IB]
    }
}

func vDSP_vneg(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = -__A[i * __IA]
    }
}

func vDSP_vsub(_ __B: UnsafePointer<Float>, _ __IB: vDSP_Stride, _ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA] - __B[i * __IB]
    }
}

func vDSP_vma(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __IB: vDSP_Stride, _ __C: UnsafePointer<Float>, _ __IC: vDSP_Stride, _ __D: UnsafeMutablePointer<Float>, _ __ID: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __D[i * __ID] = __A[i * __IA] * __B[i * __IB] + __C[i * __IC]
    }
}

func vDSP_vdiv(_ __B: UnsafePointer<Float>, _ __IB: vDSP_Stride, _ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[i * __IC] = __A[i * __IA] / __B[i * __IB]
    }
}

func vDSP_sve(_ __A: UnsafePointer<Float>, _ __I: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __N: vDSP_Length) {
    for i in 0 ..< Int(__N) {
        __C[0] += __A[i * __I]
    }
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
    for i in 0 ..< Int(__N) {
        __C[0] += __A[i * __IA] + __B[i * __IB]
    }
}

func vDSP_maxvi(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __I: UnsafeMutablePointer<vDSP_Length>, _ __N: vDSP_Length) {
    var maxI = -1
    var maxV = -Float.infinity
    
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

func vDSP_minvi(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __I: UnsafeMutablePointer<vDSP_Length>, _ __N: vDSP_Length) {
    var minI = -1
    var minV = -Float.infinity
    
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

func vDSP_vmax(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    var maxV = -Float.infinity
    
    for i in 0 ..< Int(__N) {
        let v = __A[i * __IA]
        if v > maxV {
            maxV = v
        }
    }
    
    __C[0] = maxV
}

func vDSP_vmin(_ __A: UnsafePointer<Float>, _ __IA: vDSP_Stride, _ __B: UnsafePointer<Float>, _ __IB: vDSP_Stride, _ __C: UnsafeMutablePointer<Float>, _ __IC: vDSP_Stride, _ __N: vDSP_Length) {
    var minV = -Float.infinity
    
    for i in 0 ..< Int(__N) {
        let v = __A[i * __IA]
        if v < minV {
            minV = v
        }
    }
    
    __C[0] = minV
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
    for i in 0 ..< Int(l[0]) {
        c[i] = tanhf(a[i])
    }
}

func vvsqrtf(_ c: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ l: UnsafePointer<Int32>) {
    for i in 0 ..< Int(l[0]) {
        c[i] = sqrtf(a[i])
    }
}

func vvexpf(_ c: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ l: UnsafePointer<Int32>) {
    for i in 0 ..< Int(l[0]) {
        c[i] = expf(a[i])
    }
}

func vvlogf(_ c: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ l: UnsafePointer<Int32>) {
    for i in 0 ..< Int(l[0]) {
        c[i] = logf(a[i])
    }
}

func vvsinf(_ c: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ l: UnsafePointer<Int32>) {
    for i in 0 ..< Int(l[0]) {
        c[i] = sinf(a[i])
    }
}

func vvcosf(_ c: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ l: UnsafePointer<Int32>) {
    for i in 0 ..< Int(l[0]) {
        c[i] = cosf(a[i])
    }
}

func vvtanf(_ c: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ l: UnsafePointer<Int32>) {
    for i in 0 ..< Int(l[0]) {
        c[i] = tanf(a[i])
    }
}

func vvcopysignf(_ c: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ s: UnsafePointer<Float>, _ l: UnsafePointer<Int32>) {
    for i in 0 ..< Int(l[0]) {
        c[i] = copysignf(a[i], s[i])
    }
}

// MARK: veclib Double
func vvtanh(_ c: UnsafeMutablePointer<Double>, _ a: UnsafePointer<Double>, _ l: UnsafePointer<Int32>) {
    for i in 0 ..< Int(l[0]) {
        c[i] = tanh(a[i])
    }
}

func vvsqrt(_ c: UnsafeMutablePointer<Double>, _ a: UnsafePointer<Double>, _ l: UnsafePointer<Int32>) {
    for i in 0 ..< Int(l[0]) {
        c[i] = sqrt(a[i])
    }
}

func vvexp(_ c: UnsafeMutablePointer<Double>, _ a: UnsafePointer<Double>, _ l: UnsafePointer<Int32>) {
    for i in 0 ..< Int(l[0]) {
        c[i] = exp(a[i])
    }
}

func vvlog(_ c: UnsafeMutablePointer<Double>, _ a: UnsafePointer<Double>, _ l: UnsafePointer<Int32>) {
    for i in 0 ..< Int(l[0]) {
        c[i] = log(a[i])
    }
}

func vvsin(_ c: UnsafeMutablePointer<Double>, _ a: UnsafePointer<Double>, _ l: UnsafePointer<Int32>) {
    for i in 0 ..< Int(l[0]) {
        c[i] = sin(a[i])
    }
}

func vvcos(_ c: UnsafeMutablePointer<Double>, _ a: UnsafePointer<Double>, _ l: UnsafePointer<Int32>) {
    for i in 0 ..< Int(l[0]) {
        c[i] = cos(a[i])
    }
}

func vvtan(_ c: UnsafeMutablePointer<Double>, _ a: UnsafePointer<Double>, _ l: UnsafePointer<Int32>) {
    for i in 0 ..< Int(l[0]) {
        c[i] = tan(a[i])
    }
}

func vvcopysign(_ c: UnsafeMutablePointer<Double>, _ a: UnsafePointer<Double>, _ s: UnsafePointer<Double>, _ l: UnsafePointer<Int32>) {
    for i in 0 ..< Int(l[0]) {
        c[i] = copysign(a[i], s[i])
    }
}

// MARK: cblas
typealias CBLAS_ORDER = Int8
let CblasRowMajor: CBLAS_ORDER = 101
let CblasColMajor: CBLAS_ORDER = 102

typealias CBLAS_TRANSPOSE = Int8
let CblasNoTrans: CBLAS_TRANSPOSE = 111
let CblasTrans: CBLAS_TRANSPOSE = 112

func cblas_sgemm(_ __Order: CBLAS_ORDER, _ __TransA: CBLAS_TRANSPOSE, _ __TransB: CBLAS_TRANSPOSE, _ __M: Int32, _ __N: Int32, _ __K: Int32, _ __alpha: Float, _ __A: UnsafePointer<Float>, _ __lda: Int32, _ __B: UnsafePointer<Float>, _ __ldb: Int32, _ __beta: Float, _ __C: UnsafeMutablePointer<Float>, _ __ldc: Int32) {
    // Adapted from NetLib
    @inline(__always) func A(_ I: Int32, _ J: Int32) -> Float { __A[Int((I)-1 + ((J)-1) * (__lda))] }
    @inline(__always) func B(_ I: Int32, _ J: Int32) -> Float { __B[Int((I)-1 + ((J)-1) * (__ldb))] }
    @inline(__always) func C(_ I: Int32, _ J: Int32) -> Float { __C[Int((I)-1 + ((J)-1) * (__ldc))] }
    @inline(__always) func C(_ I: Int32, _ J: Int32, _ V: Float) { __C[Int((I)-1 + ((J)-1) * (__ldc))] = V }
    
    let nota = __TransA == CblasNoTrans
    let notb = __TransB == CblasNoTrans
    let nrowa: Int32
    let ncola: Int32
    let nrowb: Int32
    
    if nota {
        nrowa = __M
        ncola = __K
    } else {
        nrowa = __K
        ncola = __M
    }
    if notb {
        nrowb = __K
    } else {
        nrowb = __N
    }

    /*     Test the input parameters. */
    var info = 0;
    if (!nota && __TransA != CblasTrans) {
        info = 1;
    } else if (!notb && __TransB != CblasTrans) {
        info = 2
    } else if (__M < 0) {
        info = 3
    } else if (__N < 0) {
        info = 4
    } else if (__K < 0) {
        info = 5
    } else if (__lda < max(1, nrowa)) {
        info = 8
    } else if (__ldb < max(1, nrowb)) {
        info = 10
    } else if (__ldc < max(1, __M)) {
        info = 13
    }
    if (info != 0) {
        fatalError("cblas_sgemm error: \(info)")
    }

    /*     Quick return if possible. */
    if (__M == 0 || __N == 0 || (__alpha == 0 || __K == 0) && __beta == 1) {
        return
    }

    /*     And if  alpha.eq.zero. */

    if (__alpha == 0) {
        if (__beta == 0) {
            for j in 1 ... __N {
                for i in 1 ... __M {
                    C(i,j,0)
                    /* L10: */
                }
                /* L20: */
            }
        } else {
            for j in 1 ... __N {
                for i in 1 ... __M {
                    C(i,j,__beta * C(i,j))
                    /* L30: */
                }
                /* L40: */
            }
        }
        return
    }

    /*     Start the operations. */

    if (notb) {
        if (nota) {
    /*           Form  C := alpha*A*B + beta*C. */
            for j in 1 ... __N {
                if (__beta == 0) {
                    for i in 1 ... __M {
                        C(i,j,0)
                    }
                } else if (__beta != 1) {
                    for i in 1 ... __M {
                        C(i,j,__beta * C(i,j))
                    }
                }
                for l in 1 ... __K {
                    if (B(l,j) != 0) {
                        let temp = __alpha * B(l,j);
                        for i in 1 ... __M {
                            C(i,j, C(i,j) + temp * A(i,l))
                        }
                    }
                }
            }
        } else {
/*           Form  C := alpha*A'*B + beta*C */
            for j in 1 ... __N {
                for i in 1 ... __M {
                    var temp: Float = 0
                    for l in 1 ... __K {
                        temp += A(l,i) * B(l,j)
                    }
                    if (__beta == 0) {
                        C(i,j,__alpha * temp)
                    } else {
                        C(i,j,__alpha * temp + __beta * C(i,j))
                    }
                }
            }
        }
    } else {
        if (nota) {
/*           Form  C := alpha*A*B' + beta*C */

            for j in 1 ... __N {
                if (__beta == 0) {
                    for i in 1 ... __M {
                        C(i,j,0)
                    }
                } else if (__beta != 1) {
                    for i in 1 ... __M {
                        C(i,j, __beta * C(i,j))
                    }
                }
                for l in 1 ... __K {
                    if (B(j,l) != 0) {
                        let temp = __alpha * B(j,l)
                        for i in 1 ... __M {
                            C(i,j, C(i,j) + temp * A(i,l))
                        }
                    }
                }
            }
        } else {
/*           Form  C := alpha*A'*B' + beta*C */

            for j in 1 ... __N {
                for i in 1 ... __M {
                    var temp: Float = 0
                    for l in 1 ... __K {
                        temp += A(l,i) * B(j,l);
                    }
                    if (__beta == 0) {
                        C(i,j,__alpha * temp)
                    } else {
                        C(i,j, __alpha * temp + __beta * C(i,j))
                    }
                }
            }
        }
    }
}

func cblas_dgemm(_ __Order: CBLAS_ORDER, _ __TransA: CBLAS_TRANSPOSE, _ __TransB: CBLAS_TRANSPOSE, _ __M: Int32, _ __N: Int32, _ __K: Int32, _ __alpha: Double, _ __A: UnsafePointer<Double>, _ __lda: Int32, _ __B: UnsafePointer<Double>, _ __ldb: Int32, _ __beta: Double, _ __C: UnsafeMutablePointer<Double>, _ __ldc: Int32) {
    // Adapted from NetLib
    @inline(__always) func A(_ I: Int32, _ J: Int32) -> Double { __A[Int((I)-1 + ((J)-1) * (__lda))] }
    @inline(__always) func B(_ I: Int32, _ J: Int32) -> Double { __B[Int((I)-1 + ((J)-1) * (__ldb))] }
    @inline(__always) func C(_ I: Int32, _ J: Int32) -> Double { __C[Int((I)-1 + ((J)-1) * (__ldc))] }
    @inline(__always) func C(_ I: Int32, _ J: Int32, _ V: Double) { __C[Int((I)-1 + ((J)-1) * (__ldc))] = V }
    
    let nota = __TransA == CblasNoTrans
    let notb = __TransB == CblasNoTrans
    let nrowa: Int32
    let ncola: Int32
    let nrowb: Int32
    
    if nota {
        nrowa = __M
        ncola = __K
    } else {
        nrowa = __K
        ncola = __M
    }
    if notb {
        nrowb = __K
    } else {
        nrowb = __N
    }

    /*     Test the input parameters. */
    var info = 0;
    if (!nota && __TransA != CblasTrans) {
        info = 1;
    } else if (!notb && __TransB != CblasTrans) {
        info = 2
    } else if (__M < 0) {
        info = 3
    } else if (__N < 0) {
        info = 4
    } else if (__K < 0) {
        info = 5
    } else if (__lda < max(1, nrowa)) {
        info = 8
    } else if (__ldb < max(1, nrowb)) {
        info = 10
    } else if (__ldc < max(1, __M)) {
        info = 13
    }
    if (info != 0) {
        fatalError("cblas_sgemm error: \(info)")
    }

    /*     Quick return if possible. */
    if (__M == 0 || __N == 0 || (__alpha == 0 || __K == 0) && __beta == 1) {
        return
    }

    /*     And if  alpha.eq.zero. */

    if (__alpha == 0) {
        if (__beta == 0) {
            for j in 1 ... __N {
                for i in 1 ... __M {
                    C(i,j,0)
                    /* L10: */
                }
                /* L20: */
            }
        } else {
            for j in 1 ... __N {
                for i in 1 ... __M {
                    C(i,j,__beta * C(i,j))
                    /* L30: */
                }
                /* L40: */
            }
        }
        return
    }

    /*     Start the operations. */

    if (notb) {
        if (nota) {
    /*           Form  C := alpha*A*B + beta*C. */
            for j in 1 ... __N {
                if (__beta == 0) {
                    for i in 1 ... __M {
                        C(i,j,0)
                    }
                } else if (__beta != 1) {
                    for i in 1 ... __M {
                        C(i,j,__beta * C(i,j))
                    }
                }
                for l in 1 ... __K {
                    if (B(l,j) != 0) {
                        let temp = __alpha * B(l,j);
                        for i in 1 ... __M {
                            C(i,j, C(i,j) + temp * A(i,l))
                        }
                    }
                }
            }
        } else {
/*           Form  C := alpha*A'*B + beta*C */
            for j in 1 ... __N {
                for i in 1 ... __M {
                    var temp: Double = 0
                    for l in 1 ... __K {
                        temp += A(l,i) * B(l,j)
                    }
                    if (__beta == 0) {
                        C(i,j,__alpha * temp)
                    } else {
                        C(i,j,__alpha * temp + __beta * C(i,j))
                    }
                }
            }
        }
    } else {
        if (nota) {
/*           Form  C := alpha*A*B' + beta*C */

            for j in 1 ... __N {
                if (__beta == 0) {
                    for i in 1 ... __M {
                        C(i,j,0)
                    }
                } else if (__beta != 1) {
                    for i in 1 ... __M {
                        C(i,j, __beta * C(i,j))
                    }
                }
                for l in 1 ... __K {
                    if (B(j,l) != 0) {
                        let temp = __alpha * B(j,l)
                        for i in 1 ... __M {
                            C(i,j, C(i,j) + temp * A(i,l))
                        }
                    }
                }
            }
        } else {
/*           Form  C := alpha*A'*B' + beta*C */

            for j in 1 ... __N {
                for i in 1 ... __M {
                    var temp: Double = 0
                    for l in 1 ... __K {
                        temp += A(l,i) * B(j,l);
                    }
                    if (__beta == 0) {
                        C(i,j,__alpha * temp)
                    } else {
                        C(i,j, __alpha * temp + __beta * C(i,j))
                    }
                }
            }
        }
    }
}


func cblas_scopy(_ __N: Int32, _ __X: UnsafePointer<Float>!, _ __incX: Int32, _ __Y: UnsafeMutablePointer<Float>!, _ __incY: Int32) {
    for i in 0 ..< __N {
        __Y[Int(i * __incY)] = __X[Int(i * __incX)]
    }
}

func cblas_dcopy(_ __N: Int32, _ __X: UnsafePointer<Double>!, _ __incX: Int32, _ __Y: UnsafeMutablePointer<Double>!, _ __incY: Int32) {
    for i in 0 ..< __N {
        __Y[Int(i * __incY)] = __X[Int(i * __incX)]
    }
}

#endif
#if canImport(Accelerate)
import Accelerate
#endif

func cblas_igemm(_ __Order: CBLAS_ORDER, _ __TransA: CBLAS_TRANSPOSE, _ __TransB: CBLAS_TRANSPOSE, _ __M: Int32, _ __N: Int32, _ __K: Int32, _ __alpha: Int32, _ __A: UnsafePointer<Int32>, _ __lda: Int32, _ __B: UnsafePointer<Int32>, _ __ldb: Int32, _ __beta: Int32, _ __C: UnsafeMutablePointer<Int32>, _ __ldc: Int32) {
    // Adapted from NetLib
    @inline(__always) func A(_ I: Int32, _ J: Int32) -> Int32 { __A[Int((I)-1 + ((J)-1) * (__lda))] }
    @inline(__always) func B(_ I: Int32, _ J: Int32) -> Int32 { __B[Int((I)-1 + ((J)-1) * (__ldb))] }
    @inline(__always) func C(_ I: Int32, _ J: Int32) -> Int32 { __C[Int((I)-1 + ((J)-1) * (__ldc))] }
    @inline(__always) func C(_ I: Int32, _ J: Int32, _ V: Int32) { __C[Int((I)-1 + ((J)-1) * (__ldc))] = V }
    
    let nota = __TransA == CblasNoTrans
    let notb = __TransB == CblasNoTrans
    let nrowa: Int32
    let ncola: Int32
    let nrowb: Int32
    
    if nota {
        nrowa = __M
        ncola = __K
    } else {
        nrowa = __K
        ncola = __M
    }
    if notb {
        nrowb = __K
    } else {
        nrowb = __N
    }

    /*     Test the input parameters. */
    var info = 0;
    if (!nota && __TransA != CblasTrans) {
        info = 1;
    } else if (!notb && __TransB != CblasTrans) {
        info = 2
    } else if (__M < 0) {
        info = 3
    } else if (__N < 0) {
        info = 4
    } else if (__K < 0) {
        info = 5
    } else if (__lda < max(1, nrowa)) {
        info = 8
    } else if (__ldb < max(1, nrowb)) {
        info = 10
    } else if (__ldc < max(1, __M)) {
        info = 13
    }
    if (info != 0) {
        fatalError("cblas_sgemm error: \(info)")
    }

    /*     Quick return if possible. */
    if (__M == 0 || __N == 0 || (__alpha == 0 || __K == 0) && __beta == 1) {
        return
    }

    /*     And if  alpha.eq.zero. */

    if (__alpha == 0) {
        if (__beta == 0) {
            for j in 1 ... __N {
                for i in 1 ... __M {
                    C(i,j,0)
                    /* L10: */
                }
                /* L20: */
            }
        } else {
            for j in 1 ... __N {
                for i in 1 ... __M {
                    C(i,j,__beta * C(i,j))
                    /* L30: */
                }
                /* L40: */
            }
        }
        return
    }

    /*     Start the operations. */

    if (notb) {
        if (nota) {
    /*           Form  C := alpha*A*B + beta*C. */
            for j in 1 ... __N {
                if (__beta == 0) {
                    for i in 1 ... __M {
                        C(i,j,0)
                    }
                } else if (__beta != 1) {
                    for i in 1 ... __M {
                        C(i,j,__beta * C(i,j))
                    }
                }
                for l in 1 ... __K {
                    if (B(l,j) != 0) {
                        let temp = __alpha * B(l,j);
                        for i in 1 ... __M {
                            C(i,j, C(i,j) + temp * A(i,l))
                        }
                    }
                }
            }
        } else {
/*           Form  C := alpha*A'*B + beta*C */
            for j in 1 ... __N {
                for i in 1 ... __M {
                    var temp: Int32 = 0
                    for l in 1 ... __K {
                        temp += A(l,i) * B(l,j)
                    }
                    if (__beta == 0) {
                        C(i,j,__alpha * temp)
                    } else {
                        C(i,j,__alpha * temp + __beta * C(i,j))
                    }
                }
            }
        }
    } else {
        if (nota) {
/*           Form  C := alpha*A*B' + beta*C */

            for j in 1 ... __N {
                if (__beta == 0) {
                    for i in 1 ... __M {
                        C(i,j,0)
                    }
                } else if (__beta != 1) {
                    for i in 1 ... __M {
                        C(i,j, __beta * C(i,j))
                    }
                }
                for l in 1 ... __K {
                    if (B(j,l) != 0) {
                        let temp = __alpha * B(j,l)
                        for i in 1 ... __M {
                            C(i,j, C(i,j) + temp * A(i,l))
                        }
                    }
                }
            }
        } else {
/*           Form  C := alpha*A'*B' + beta*C */

            for j in 1 ... __N {
                for i in 1 ... __M {
                    var temp: Int32 = 0
                    for l in 1 ... __K {
                        temp += A(l,i) * B(j,l);
                    }
                    if (__beta == 0) {
                        C(i,j,__alpha * temp)
                    } else {
                        C(i,j, __alpha * temp + __beta * C(i,j))
                    }
                }
            }
        }
    }
}


func cblas_icopy(_ __N: Int32, _ __X: UnsafePointer<Int32>!, _ __incX: Int32, _ __Y: UnsafeMutablePointer<Int32>!, _ __incY: Int32) {
    for i in 0 ..< __N {
        __Y[Int(i * __incY)] = __X[Int(i * __incX)]
    }
}
