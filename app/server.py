from flask import Flask,request,jsonify
from flask_cors import CORS
import numpy as np
import timeit
import json

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def get_instance_taillard(C,x,y) :
    C = C.replace("\n", "")
    C = C.split(" ")
    C = list(filter(lambda a: a != '' , C))
    return np.array(C,dtype=int).reshape(x,y)

def get_start_time(pt,order) :
    machines = pt.shape[0]
    jobs = pt.shape[1]
    start = np.empty_like(pt)
    end = np.empty_like(pt)

    end[0][0] = pt[0,order[0] - 1]
    start[0][0] = 0

    for j in range(1,jobs) :
        start[0][j] = end[0][j - 1] 
        end[0][j] = start[0][j] + pt[0,order[j] - 1] 

    for i in range(1,machines) :
        start[i][0] = end[i-1][0]
        end[i][0] = start[i][0] + pt[i,order[0] - 1]
        for j in range(1,jobs) :
            start[i][j] = max(end[i-1][j],end[i][j-1])
            end[i][j] = start[i][j] + pt[i,order[j] - 1]
    res = []
    for i in range(machines) :
        for j in range(jobs) :
            res.append(["M"+str(i+1),"J"+ str(order[j]), int(start[i][j]) * 1000,int(end[i][j])* 1000])
    return res

# print(get_start_time(np.array([[1,2],[3,4]]),[1,2]))
def taillard_to_file(C,filename) :
    file = open(filename,"w")
    l0 = str(C.shape[0])+"  "+str(C.shape[1])+"\n"
    file.write(l0)
    for i in range(C.shape[0]):
        l = ""
        for j in range(C.shape[1]):
            if j!= 0:
                wr ="  "+str(j)+"  "+str(C[i,j])
            else:
                wr = str(j)+"  "+str(C[i,j])

            l+=wr
        l+="\n"
        file.write(l)
    file.close()

@app.route("/",methods=["GET"])
def home() :
    return {"msg":"test api working!"}
    
@app.route("/branch-and-bound",methods=["POST"])
def branchAndBound():

    from app.branchAndBound.BB import BrandAndBound

    try :
        processing_time = request.json['ptMatrice']
        m = request.json['nbMachine']
        j = request.json['nbJob']
        C = get_instance_taillard(processing_time,int(m),int(j))
        instance = BrandAndBound(C)
        start = timeit.default_timer()
        instance.run()
        end = timeit.default_timer()
        data = get_start_time(C,instance.ordre_opt)

        return {"instance":C.tolist(),"jobs_order":instance.ordre_opt,"cmax":instance.M,"time":end - start,'data':data}

    except Exception as e :
        print(e)
        return {'error':  'Nous pouvons pas traiter votre instance, vérifier ses dimensions.'}

@app.route("/heuristic-neh",methods=["POST"])
def heuristicNeh():

    from app.heuristics.Neh import FSP_NEH

    try :
        processing_time = request.json['ptMatrice']
        m = request.json['nbMachine']
        j = request.json['nbJob']
        C = get_instance_taillard(processing_time,int(m),int(j))

        instance = FSP_NEH(C)
        start = timeit.default_timer()
        instance.run()
        end = timeit.default_timer()
        
        data = get_start_time(C,instance.ordre_courant)

        return {"instance":C.tolist(),"jobs_order":instance.ordre_courant,"cmax":instance.cmax,"time":end - start,'data':data}

    except Exception as e :
        print(e)
        return {'error':  'Nous pouvons pas traiter votre instance, vérifier ses dimensions.'}

@app.route("/heuristic-cds",methods=["POST"])
def heuristicCds():

    from app.heuristics.Cds import DCS
    try :
        processing_time = request.json['ptMatrice']
        m = request.json['nbMachine']
        j = request.json['nbJob']
        C = get_instance_taillard(processing_time,int(m),int(j))

        instance = DCS(C.T)
        start = timeit.default_timer()
        instance.run()
        end = timeit.default_timer()

        data = get_start_time(C,instance.solution[0])

        return {"instance":C.tolist(),"jobs_order":list(instance.solution[0]),"cmax":instance.solution[1],"time":end - start,'data':data}

    except Exception as e :
        print(e)
        return {'error':  'Nous pouvons pas traiter votre instance, vérifier ses dimensions.'}

@app.route("/heuristic-palmer",methods=["POST"])
def heuristicPalmer():

    from app.heuristics.Palmer import PalmerHeuristic
    try :
        processing_time = request.json['ptMatrice']
        m = int(request.json['nbMachine'])
        j = int(request.json['nbJob'])
        C = get_instance_taillard(processing_time,m,j)

        instance = PalmerHeuristic(C,m,j)
        start = timeit.default_timer()
        instance.run()
        end = timeit.default_timer()
        data = get_start_time(C,instance.ordre_opt)

        return {"instance":C.tolist(),"jobs_order": instance.ordre_opt.tolist(),"cmax":instance.M,"time":end - start,'data':data}

    except Exception as e :
        print(e)
        return {'error':  'Nous pouvons pas traiter votre instance, vérifier ses dimensions.'}

@app.route("/metaheuristic-tabu",methods=["POST"])
def metaheuristicTabu():

    from app.metaheuristics.Tabu import Tabu

    try :

        processing_time = request.json['ptMatrice']
        m = int(request.json['nbMachine'])
        j = int(request.json['nbJob'])
        tabutenure = int(request.json['tabutenure'])
        niter = int(request.json['niter'])
        C = get_instance_taillard(processing_time,m,j)


        start = timeit.default_timer()
        instance = Tabu(C,m,j,tabu_tenure=tabutenure,niter=niter,Verbose=False)
        end = timeit.default_timer()

        data = get_start_time(C,instance.Best_solution)

        return {"instance":C.tolist(),"jobs_order": instance.Best_solution,"cmax":instance.Best_objvalue,"time":end - start,'data':data}

    except Exception as e :
        print(e)
        return {'error':  'Nous pouvons pas traiter votre instance, vérifier ses dimensions.'}

@app.route("/metaheuristic-gvns",methods=["POST"])
def metaheuristicGvns():

    from app.metaheuristics.Gvns import GVNS

    try :
        
        processing_time = request.json['ptMatrice']
        m = int(request.json['nbMachine'])
        j = int(request.json['nbJob'])
        tabutenure = int(request.json['tabutenure'])
        niter = int(request.json['niter'])
        C = get_instance_taillard(processing_time,m,j)

        instance = GVNS(C,m,j, tabu_tenure=tabutenure,niter=niter)
        start = timeit.default_timer()
        instance.run()
        end = timeit.default_timer()
        data = get_start_time(C,instance.best_solution)

        return {"instance":C.tolist(),"jobs_order": instance.best_solution,"cmax":instance.best_solution_cmax,"time":end - start,'data':data}

    except Exception as e :
        print(e)
        return {'error':  'Nous pouvons pas traiter votre instance, vérifier ses dimensions.'}

@app.route("/metaheuristic-ga",methods=["POST"])
def metaheuristicGA():

    from app.metaheuristics.GA import run_ga

    try :
        
        processing_time = request.json['ptMatrice']
        niter = int(request.json['niter'])
        npop = int(request.json['npop'])
        crossover_porb = float(request.json['crossover_porb'])
        mutation_porb = float(request.json['mutation_porb'])
        method = request.json['method']
        nparent = int(request.json['nparent'])
        m = int(request.json['nbMachine'])
        j = int(request.json['nbJob'])
        C = get_instance_taillard(processing_time,m,j)


        taillard_to_file(C.T,"data.txt")
        instance = run_ga(niter, npop, crossover_porb, mutation_porb, method, nparent)
        jobs_order =  np.array(instance["sequence"]).tolist()
        data = get_start_time(C,jobs_order)

        return {"instance":C.tolist(),"jobs_order":jobs_order,"cmax":instance["Cmax"],"time": instance["execTime"],'data':data}

    except Exception as e :
        print(e)
        return {'error':  'Nous pouvons pas traiter votre instance, vérifier ses dimensions.'}

@app.route("/hyperheuristic",methods=["POST"])
def Hyperheuristic():

    from app.hyperHeuristics.HhQl import QLHH
    from app.heuristics.Palmer import PalmerHeuristic

    try :
        
        processing_time = request.json['ptMatrice']
        niter = int(request.json['niter'])
        nepis = int(request.json['nepis'])
        prob = float(request.json['prob'])
        m = int(request.json['nbMachine'])
        j = int(request.json['nbJob'])
        C = get_instance_taillard(processing_time,m,j)


        instance = PalmerHeuristic(C,m,j)
        instance.run()
        initial_sol,initial_fun = instance.ordre_opt,instance.M
        start = timeit.default_timer()
        best_sol, best_fun = QLHH(C,initial_sol,initial_fun,maxiter=niter,n_episodes=nepis,ep=prob)
        end = timeit.default_timer()

        data = get_start_time(C,best_sol)
        best_fun = int(best_fun)
        C = C.tolist()
        return json.dumps({"instance":C,"jobs_order":best_sol,"cmax":best_fun,"time": end - start,'data':data},cls=NpEncoder)

    except Exception as e :
        print(e)
        return {'error':  'Nous pouvons pas traiter votre instance, vérifier ses dimensions.'}



@app.route("/comparaison-heuristics",methods=["POST"])
def comparaisonHeuristics():

    from app.heuristics.Neh import FSP_NEH
    from app.heuristics.Palmer import PalmerHeuristic
    from app.heuristics.Cds import DCS

    try :
        processing_time = request.json['ptMatrice']
        m = int(request.json['nbMachine'])
        j = int(request.json['nbJob'])
        C = get_instance_taillard(processing_time,m,j)

        instance_neh = FSP_NEH(C)
        start_neh = timeit.default_timer()
        instance_neh.run()
        end_neh = timeit.default_timer()

        instance_palmer = PalmerHeuristic(C,m,j)
        start_palmer = timeit.default_timer()
        instance_palmer.run()
        end_palmer = timeit.default_timer()
        
        instance_cds = DCS(C.T)
        start_cds = timeit.default_timer()
        instance_cds.run()
        end_cds = timeit.default_timer()

        return jsonify([["NEH",instance_neh.cmax,end_neh-start_neh],
                ["CDS",instance_cds.solution[1],end_cds-start_cds],
                ["Palmer",instance_palmer.M,end_palmer-start_palmer]])

    except Exception as e :
        print(e)
        return {'error':  'Nous pouvons pas traiter votre instance, vérifier ses dimensions.'}

if __name__ == "__main__" :
    app.run(debug=True)
