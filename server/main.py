import random
import warnings

from flask import Flask, abort, request
from flask_cors import CORS, cross_origin
from os.path import join

from model.env import Environment
from model.agent import DDPGAgent, FCFSAgent

warnings.filterwarnings('ignore')

app = Flask(__name__)
cors = CORS(app, resources={r"/foo": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/generate', methods=['GET'])
@cross_origin(origin='*', headers=['Content-Type','Authorization'])
def generate():
    seed = random.randint(0, 1000)
    env = Environment(max_power=100, positions=20, is_train=False, seed=seed)
    env.reset()

    data = env.to_json()
    data['algorithm'] = 'ddpg'

    return data

@app.route('/simulate', methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type','Authorization'])
def simulate():
    data = request.json
    algorithm = data['algorithm']
    
    env = Environment.generate_env_from_json(data)

    if data['algorithm'] == 'ddpg':
        agent = DDPGAgent(n_actions=20, input_dims=[120])
        agent.load_agent(join('model', 'trained_model'))
        actions = agent.predict(observation=env.get_observation())
        actions = DDPGAgent.normalize_actions(actions, env.max_power)
    elif data['algorithm'] == 'fcfs':
        agent = FCFSAgent()
        actions = agent.predict(observation=env.get_observation(),
            max_power=env.max_power)
    else:
        abort(404)


    _ = env.step(charging_power_per_vehicle=actions)

    data = env.to_json()
    data['algorithm'] = algorithm
    
    return data

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader=False)