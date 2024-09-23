import os
for i in range(1):
    os.system("cp -f template.sh multirun.sh")
    os.system("echo ""your new command to run the program"" >>multirun.sh")
    os.system("sbatch multirun.sh")
