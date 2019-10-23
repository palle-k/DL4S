Pod::Spec.new do |s|
    s.name = 'DL4S'
    s.version = '0.2.0'
    s.license = 'MIT'
    s.summary = 'C Library for cross platform vectorized math'
    s.homepage = 'https://github.com/palle-k/DL4S'
    s.authors = 'Palle Klewitz'
    s.source = { :git => 'https://github.com/palle-k/DL4S.git', :tag => s.version }

    s.swift_version = '5.1'
    s.default_subspec = 'Core'

    s.osx.deployment_target = '10.15'
    s.ios.deployment_target = '13.0'
    s.tvos.deployment_target = '13.0'
    s.watchos.deployment_target = '6.0'
    
    s.subspec 'Core' do |core|
        core.source_files = 'Sources/DL4S/**/*.swift'
        core.dependency 'DL4S/Lib'
    end
    
    s.subspec 'Lib' do |lib|
        lib.source_files = 'Sources/DL4SLib/**/*.{c,h}'
        lib.source_files = 'Sources/DL4SLib/include/*.h'
    end

end
