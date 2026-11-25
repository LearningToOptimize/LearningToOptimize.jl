### **Reading from and Compressing Arrow Files**

#### **Introduction**
The package provides tools to work with Arrow files efficiently, allowing users to store and retrieve large datasets efficiently.

```julia
output_file = joinpath(save_path, "$(case_name)_output_$(batch_id)")
recorder = Recorder{filetype}(output_file)
successful_solves = solve_batch(problem_iterator, recorder)
```

#### **Compressing Arrow Files**

Since appending data to Arrow files is slow and inefficient, 
each instance of data is stored in a separate file. Therefore, in this case, the output files will look like this:

```
<case_name>_output_<batch_id>_<instance_1_id>.arrow
<case_name>_output_<batch_id>_<instance_2_id>.arrow
...
<case_name>_output_<batch_id>_<instance_n_id>.arrow
```

`LearningToOptimize.jl` supports compressing batches of Arrow files for streamlined storage and retrieval.

Use the `LearningToOptimize.compress_batch_arrow` function to compress a batch of Arrow files into a single file. This reduces disk usage and simplifies file management.

**Function Signature**:
```julia
LearningToOptimize.compress_batch_arrow(
    save_path,
    case_name;
    keyword_all = "output",
    batch_id = string(batch_id),
    # keyword_any = [string(batch_id)]
)
```

- **Arguments**:
  - `save_path`: Path to save the compressed file.
  - `case_name`: Name of the case or batch.
  - `keyword_all`: Filter files containing this keyword (default: `"output"`).
  - `batch_id`: Identifier for the batch of files.
  - `keyword_any`: Array of keywords to further filter files.

The compressed file will be saved as `<case_name>_output_<batch_id>.arrow`.


#### **Reading Arrow Files**
Arrow files can be read using Julia’s Arrow library, which provides a tabular interface for data access.

**Example**:
```julia
using Arrow

# Read compressed Arrow file
data = Arrow.Table("<case_name>_output_<batch_id>.arrow")

# Access data as a DataFrame
using DataFrames
df = DataFrame(data)

println("DataFrame content:")
println(df)
```
