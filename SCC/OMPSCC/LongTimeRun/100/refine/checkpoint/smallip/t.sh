
echo "start time is"
date
start_time=$(date +%s)
sleep 1
end_time=$(date +%s)
execution_time=$(( end_time - start_time ))
echo "execution time is $execution_time"
