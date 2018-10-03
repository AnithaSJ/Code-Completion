# Code-Completion

### Problem/ Project Description:

The goal is to implement a neural network based code completion approach. Given the partial program, the trained model should be able to suggest the missing code at the specified location. For this, the model is trained using 800 JavaScript programs. However, the same approach can be used for other programming languages.

**Model used:** Long Short-Term Memory (LSTM).

**Tools Used:** TensorFlow and TFLearn

**Language used:** Python

**Try it Yourself:** You can download and execute the python file after installing all the tools used.

**Note:** Cloud based GPU was used. You can find out more about it ["here"](https://www.floydhub.com/pricing).

### Quick Overview:

**Data Representation/Pre-processing:** The JavaScript programs are represented as a sequence of tokens. Each token has two properties-- Type and Value. An example is shown below. 

JavaScript code snippet:
```
this.assertTrue(isCrossDomain("http://cross.domain"), "cross");
```
Token representation:
```
{"type": "Keyword","value": "this"},
{"type": "Punctuator","value": "."},
{"type": "Identifier","value": "assertTrue"},
{"type": "Punctuator","value": "("},
{"type": "Identifier","value": "isCrossDomain"},
{"type": "Punctuator","value": "("},
{"type": "String","value": "\"http://cross.domain\""},
{"type": "Punctuator","value": ")"},
{"type": "Punctuator","value": ","},
{"type": "String","value": "\"cross\""},
{"type": "Punctuator","value": ")"},
{"type": "Punctuator","value": ";"}
  ```
This is further simplified using the following method:
```
def simplify_token(token):
    if token["type"] == "Identifier":
        token["value"] = "ID"
    elif token["type"] == "String":
        token["value"] = "\"STR\""
    elif token["type"] == "RegularExpression":
        token["value"] = "/REGEXP/"
    elif token["type"] == "Numeric":
        token["value"] = "5"
```
**Results** A program is divided into three sections-- Prefix, Expected and Suffix. Consider the code snippet ```this.assertTrue(isCrossDomain("http://cross.domain"), "cross");```. For the purpose of convenience, let's assume the missing token size to be one i.e., only one token can be missing. Here, let's say that the missing token is ```(``` after ```assertTrue```. This means that ```this.assertTrue``` is the prefix of the missing/expected token ```(``` and the code following it is the suffix ```isCrossDomain("http://cross.domain"), "cross");```. 

The model is queried with the prefix and the suffix of the missing section(s)/token(s). The trained model in turn provides the sequence of token(s) for the missing section(s). So far, the following results are obtained for the maximum size of missing tokens of two i.e., the maximum length of consecutive missing tokens is two.

<img src="https://github.com/Meghana-Meghana/Code-Completion-using-Deep-Learning/blob/master/level%203/Results.png" width="600">

**Future Scope**
  * The maximum size of missing tokens is two. It can be generalized to larger missing tokens.
  * A hybrid model of LSTM and N-gram can be considered for training.
  
**Reading Material**

["Code Completion with Statistical Language Models"](http://www.cs.technion.ac.il/~yahave/papers/pldi14-statistical.pdf)

**PS** This project was done as part of a [course](http://software-lab.org/teaching/summer2018/asdl/) in the university.

 
 
 



