@echo off
 
set "targetPath=output"
 
if "%1"=="" (
    for /f "delims=" %%A in ('dir /ad-h /tw /b "%targetPath%"') do (
        set "targetFolder=%%A"
    )
) else (
    set "targetFolder=%1"
)
python render.py -m "output\%targetFolder%"
python metrics.py -m "output\%targetFolder%"
