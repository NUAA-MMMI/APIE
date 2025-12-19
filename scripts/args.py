import argparse


class UncertaintyArgs():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--model", type=str, required=True, help="Path to the model file or model name")
        self.parser.add_argument("--client", type=str, default="ollama", help="client to use (default: ollama)")
        self.parser.add_argument("--inputFile", type=str, required=True, help="Path to the input Directory")
        self.parser.add_argument("--schema", type=str, required=True, help="Path to the schema file")

        self.parser.add_argument("--shotSize", type=int)
        self.parser.add_argument("--responseSize", type=int)
        self.parser.add_argument("--pollSize", type=int)
        self.parser.add_argument("--method", type=str, default="ZSL", help="Generation method to use (default: ZSL)")
        self.parser.add_argument("--uncertaintyFile", type=str, default=None, help="Path to the uncertainty file")
    def parse(self):
        return self.parser.parse_args()
    
class InferenceArgs():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--inputFile", type=str, required=True, help="Path to the input Directory")
        self.parser.add_argument("--schema", type=str, required=True, help="Path to the schema file")
        self.parser.add_argument("--outputFile", type=str, required=True, help="Path to the output Directory")
        
        self.parser.add_argument("--model", type=str, required=True, help="Path to the model file or model name")
        self.parser.add_argument("--client", type=str, default="ollama", help="client to use (default: ollama)")

        self.parser.add_argument("--method", type=str, default="ZSL", help="Generation method to use (default: default)")
        self.parser.add_argument("--shotSize", type=int, help="Temperature for generation")
        self.parser.add_argument("--pollSize", type=int, help="Temperature for generation")
        self.parser.add_argument("--uncertaintyFile", type=str, help="Path to the uncertainty file")

        self.parser.add_argument("--alpha", type=float, default=0.1, help="Alpha value for uncertainty calculation (default: 0.5)")
        self.parser.add_argument("--beta", type=float, default=0.1, help="Beta value for uncertainty calculation (default: 0.5)")
        self.parser.add_argument("--gama", type=float, default=0.8, help="Gama value for uncertainty calculation (default: 0.5)")
        self.parser.add_argument("--frontArgs",type=float,default=0.1, help="Front arguments for the model (default: 0.5)")

    def parse(self):
        return self.parser.parse_args()
    
class EvaluationArgs():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        self.parser.add_argument("--inputFile", type=str, required=True, help="Path to the input file")
        self.parser.add_argument("--schema", type=str, required=True, help="Path to the schema file")
        self.parser.add_argument("--outputFile", type=str, required=True, help="Path to the output Directory")
        self.parser.add_argument("--shotSize", type=int, help="Number of shots to use for evaluation")
        self.parser.add_argument("--recordFile", type=str, help="Path to the record file for uncertainty calculation")
        self.parser.add_argument("--uncertaintyFile", type=str, help="Path to the uncertainty file")
        self.parser.add_argument("--alpha", type=float, default=0.5, help="Alpha value for uncertainty calculation (default: 0.5)")
        self.parser.add_argument("--beta", type=float, default=0.5, help="Beta value for uncertainty calculation (default: 0.5)")
        self.parser.add_argument("--gama", type=float, default=0.5, help="Gama value for uncertainty calculation (default: 0.5)")

    def parse(self):
        return self.parser.parse_args()