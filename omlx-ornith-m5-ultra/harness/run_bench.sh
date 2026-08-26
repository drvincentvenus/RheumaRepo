#!/bin/zsh
# Sequential benchmark: models x clients x tasks x reps
set -u
B=~/Documents/omlx_bench; cd $B
CODE='In the current directory create fizzbuzz.py that prints FizzBuzz for 1..30, run it with python3, then reply with the last 3 lines of its output.'
TEXT='Explain in about 300 words how expert routing works in a Mixture-of-Experts transformer. Do not use tools.'
for MODEL in ornith-4bit ornith-6bit; do
  # unload other model by simply using this one; oMLX LRU handles it
  for CLIENT in hermes opencode; do
    for TASK in code text; do
      for REP in 1 2 3; do
        TAG="$MODEL|$CLIENT|$TASK|rep$REP"; echo "$TAG" > TAG
        W=$B/work/${MODEL}_${CLIENT}_${TASK}_$REP; rm -rf $W; mkdir -p $W; cd $W
        P=$CODE; [[ $TASK == text ]] && P=$TEXT
        echo "### $(date +%T) $TAG"
        if [[ $CLIENT == hermes ]]; then
          hermes -z "$P" --provider omlx -m $MODEL > out.txt 2>&1
        else
          opencode run -m omlx/$MODEL "$P" > out.txt 2>&1
        fi
        echo "exit=$? lines=$(wc -l < out.txt)"; cd $B
      done
    done
  done
done
echo "### DONE $(date +%T)"
