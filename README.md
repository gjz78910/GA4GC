# GreenerAgent

## General ideas:
Coding agents’ idea:
Plain:
-	1st Open-source mistral (Ollama)
-	Benchmark: the LLM generates the patch and the benchmark runs everything in Docker to evaluate the performance: Result -> Fail / Optimised
Paid version:
-	Check the results in the paid version, we can maybe compare the quality with respect to the free one
GIN:
-	Code improvement. 
Green:
-	Selection of the smallest LLMs to get the same results.
-	Python for carbon print measurement
Search Component:
-	Parameter fine-tuning for the LLMs

## Steps for the Paper

1)	Preparing the Benchmark: Fitness is runtime + extend the code carbon from Python
2)	Preparing the Open LLMs (Container): GA optimizing the carbon print and runtime. Maybe use different model sides. Maybe focus on 2 small models. Keep track of the paper cost/foot print.  [Potentially use Qwen2.5/Coder with different sizes].
3)	Preparing the Mistral Code: measure the quality with the optimized version and see if it gets a better runtime or the same.


