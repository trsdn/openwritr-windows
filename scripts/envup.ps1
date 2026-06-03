$env:Path = "C:\Program Files\LLVM\bin;$env:USERPROFILE\.cargo\bin;$env:Path"
$vc = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat'
$tmp = "$env:TEMP\vcenv.txt"
cmd /c "call `"$vc`" arm64 && set PATH&& set INCLUDE&& set LIB&& set LIBPATH" 2>&1 | Set-Content $tmp
foreach ($line in (Get-Content $tmp)) {
    if ($line -match "^(PATH|INCLUDE|LIB|LIBPATH)=(.*)$") {
        Set-Item "Env:$($Matches[1])" $Matches[2]
    }
}
$env:Path = "C:\Program Files\LLVM\bin;$env:USERPROFILE\.cargo\bin;$env:Path"
Remove-Item $tmp -EA SilentlyContinue
