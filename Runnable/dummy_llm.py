#dummy llm class generated like actual class behaves 
import random
class Dummyllm:
    def __init__ (self):
        print('llm created')

    def predict(self,prompt):
        response_list = [
            "delhi is the capital of india",
            "Mumbai is the big state of india",
            "Gateway of india is in the mumbai"
            ]
        return {'response': random.choice(response_list)}



class Prompt_template:
    def __init__ (self, template,input_variables):
        self.template = template 
        self.input_variabled = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict)

template = Prompt_template(
    template = 'write a {length} poem about topic {topic}',
    input_variables=['length','topic']
)

template.format({'length': 'short', 'topic': 'india'})



class NakliChain:
    def __init__(self,llm,prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):
       final_prompt =  self.prompt.format(input_dict)
       result = self.llm.predict(final_prompt)

       return result['response']