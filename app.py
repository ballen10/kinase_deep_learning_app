from flask import Flask, render_template, request, make_response, send_file
from wtforms import Form, TextAreaField, validators
import pickle, gzip, csv
import sqlite3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from deepchem.datasets import Dataset
from deepchem.models.tensorflow_models import TensorflowModel
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.embed import components
from bokeh.palettes import Plasma256
import pandas as pd

app = Flask(__name__)
Plasma256.extend(Plasma256[::-1])
cur_dir = os.path.dirname(__file__)
data_dir = os.path.join(cur_dir,'data')
base_dir = '/data/ballen/ML/kinaseDeepLearningAllKinase_081516'
test_dir = os.path.join(base_dir,'test_dataset_random')
kinase_tasks = Dataset(test_dir,reload=True).get_task_names()
kinase_task_types = {task: 'classification' for task in kinase_tasks}
params_dict = {"activation": "relu",
                "momentum": .9,
                "batch_size": 128,
                "init": "glorot_uniform",
                "data_shape": (1024,),
                "learning_rate": 1e-3,
                "decay": 1e-6,
                "nb_hidden": (2000,500), 
                "nb_epoch": 100,
                "nesterov": False,
                "dropouts": (.5,.5),
                "nb_layers": 2,
                "batchnorm": False,
                "layer_sizes": (2000,500),
                "weight_init_stddevs": (.1,.1),
                "bias_init_consts": (1.,1.),
                "num_classes": 2,
                "penalty": 0., 
                "optimizer": "sgd",
                "num_classification_tasks": len(kinase_task_types)
                  }
model_dir = os.path.join(base_dir,'model_2000x500_128_allKinase_081516')
model = TensorflowModel(kinase_tasks,kinase_task_types,params_dict,model_dir,tf_class=TensorflowMultiTaskClassifier,
                       verbosity='high')

def classify(document):
	doc = document.strip().split('\r\n')
	mol = [Chem.MolFromSmiles(x) for x in doc if x is not None]
	fp = [AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=1024) for x in mol if x is not None]
	fp = np.array(fp)
	if len(fp.shape) == 1:
	    fp = np.reshape(fp,(1,-1))
	o = np.ones((fp.shape[0],len(kinase_tasks)))
	d = Dataset.from_numpy(data_dir,fp,o,tasks=kinase_tasks)
	y = np.squeeze(np.delete(model.predict_proba(d),0,2))
	if len(y.shape) == 1:
	    y = np.reshape(y, (1,-1))
	yy = pd.DataFrame(y)
	yy.columns = kinase_tasks
	yy = yy.T
	yy.columns = doc
	yy.index.name = 'kinase'
	yy.to_csv(os.path.join(data_dir,'pred.csv'))
	return doc,yy

def heatmap(doc,y):
	source = ColumnDataSource(data=dict(
	    xname=doc,
	    yname=kinase_tasks,
	    colors=Plasma256,
	    count=y))
	p = figure(title="Kinase Prediction Heatmap", x_axis_location="above", tools="hover, save",
		x_range=doc[1:], y_range=kinase_tasks,plot_width=800,plot_height=800)
	p.grid.grid_line_color=None
	p.axis.axis_line_color=None
	p.axis.major_tick_line_color=None

	p.rect('xname','yname',0.9,0.9,source=source,
		color='colors',line_color=None,alpha='count',
		hover_line_color='black',hover_color='colors')

	p.select_one(HoverTool).tooltips = [
	    ('names', '@yname, @xname'),
	    ('count', '@count'),]
	script,div=components(p)
	return script,div

def plot(y):
	kt = kinase_tasks[::-1]
	s = y.stack().reset_index()
	s.columns = ['kinase','compound','value']
	s.sort_values('kinase',axis=0,ascending=False,inplace=True)
	source = ColumnDataSource(data=s)
	hover = HoverTool(tooltips=[("compound","@compound"),("kinase","@kinase"),("value","@value")])
	dot = figure(title="Kinase Predictions", tools=[hover], y_range=kt, x_range=[0,1], plot_height=4000, plot_width=1500)
	dot.segment(0, 'kinase', 'value', 'kinase', source=source, line_width=2, line_color=Plasma256,)
	dot.circle('value', 'kinase', source=source, size=15, fill_color=Plasma256, line_color='black', line_width=3,)	
	dot.xaxis.axis_label = "Predicted Probability of Kinase Inhibition"
	dot.yaxis.axis_label = "Kinase Target"
	script, div = components(dot)
	return script, div

class ReviewForm(Form):
	smiles = TextAreaField('',
		[validators.DataRequired(),
		validators.length(min=1)])

@app.route('/')
def index():
	form = ReviewForm(request.form)
	return render_template('reviewform.html', form=form)

@app.route('/results', methods = ['POST'])
def results():
	form = ReviewForm(request.form)
	if request.method == 'POST' and form.validate():
		smi = request.form['smiles']
		doc, yy = classify(smi)
		script,div = plot(yy)
		return render_template('results.html', content=doc,
			prediction=yy,script=script,div=div)
#, probability=np.round(y*100,2).astype(np.int))
	return render_template('reviewform.html', form=form)

@app.route('/download')
def download():
	try:
	    return send_file(os.path.join(data_dir,'pred.csv'),attachment_filename='data.csv',as_attachment=True)
	except Exception as e:
	    return str(e)

if __name__ == '__main__':
	app.run(host='0.0.0.0',debug=True)
