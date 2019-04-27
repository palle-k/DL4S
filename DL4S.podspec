Pod::Spec.new do |s|
    s.name = 'DL4S'
    s.version = '0.1.2'
    s.license = 'MIT'
    s.summary = 'Dynamic neural networks based on reverse mode automatic differentiation'
    s.homepage = 'https://github.com/palle-k/DL4S'
    s.authors = 'Palle Klewitz'
    s.source = { :git => 'https://github.com/palle-k/DL4S.git', :tag => s.version }

    s.swift_version = '5.0'

    s.osx.deployment_target = '10.12'
    s.ios.deployment_target = '11.0'
    s.tvos.deployment_target = '11.0'

    s.source_files = 'Sources/DL4S/**/*.swift'
end
