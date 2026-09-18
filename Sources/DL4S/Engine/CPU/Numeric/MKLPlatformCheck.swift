//
//  MKLPlatformCheck.swift
//  DL4S
//
//  Created by Palle Klewitz on 10.09.26.
//  Copyright (c) 2026 - Palle Klewitz
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

// Prevent oneAPI traits / enable-flags from being active on unsupported platforms.
// The DocC plugin builds the symbol graph with all package traits enabled, so a documentation
// build passes -DDL4S_SKIP_MKL_PLATFORM_CHECK to compile the Accelerate or generic variant instead.
#if (MKL_UNSUPPORTED_PLATFORM || (MKL_ENABLE && !(os(Linux) && arch(x86_64)))) && !DL4S_SKIP_MKL_PLATFORM_CHECK
#error("Intel oneAPI MKL and IPP are unsupported on this platform. Use Accelerate on Apple platforms.")
#endif
