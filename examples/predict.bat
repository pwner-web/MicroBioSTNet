@echo off

cd ..
REM Start process for subject A Saliva
echo start at %date% %time%
python predict.py -i data/A_Saliva_abun.csv -s "subject A Saliva" --num_timesteps_output 1 --enable-cuda
echo subject A Saliva end at %date% %time%

REM Start process for subject A Stool
python predict.py -i data/A_Stool_abun.csv -s "subject A Stool" --num_timesteps_output 1 --enable-cuda
echo subject A Stool end at %date% %time%

REM Start process for subject M3 Stool
python predict.py -i data/M3_gut_abun.csv -s "subject M3 Stool" --num_timesteps_output 1 --enable-cuda
echo subject M3 Stool end at %date% %time%

REM Start process for subject M3 Saliva
python predict.py -i data/M3_tongue_abun.csv -s "subject M3 Saliva" --num_timesteps_output 1 --enable-cuda
echo subject M3 Saliva end at %date% %time%

REM Start process for subject B Stool
python predict.py -i data/B_Stool_abun.csv -s "subject B Stool" --num_timesteps_output 1 --enable-cuda
echo subject B Stool end at %date% %time%

REM Start process for subject F4 Stool
python predict.py -i data/F4_gut_abun.csv -s "subject F4 Stool" --num_timesteps_output 1 --enable-cuda
echo subject F4 Stool end at %date% %time%

REM Start process for subject F4 Saliva
python predict.py -i data/F4_tongue_abun.csv -s "subject F4 Saliva" --num_timesteps_output 1 --enable-cuda
echo subject F4 Saliva end at %date% %time%
