"""Deterministic CPU checks for the revised scoring and metric definitions."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol
import csv
import r1_experiments as r
import numpy as np
import torch
from controllers import LinearMPC, FedUMPCController


class FixedPredictor:
    def predict_horizon(self, states, actions):
        states=np.asarray(states);actions=np.asarray(actions)
        mean=np.c_[.1*states[:,2]*np.cos(states[:,3]),
                   .1*states[:,2]*np.sin(states[:,3]),
                   .1*actions[:,1],.02*actions[:,0]]
        return mean,np.full(len(states),.02)
    def predict_uncertainty(self,state,action):
        m,v=self.predict_horizon(np.asarray(state)[None],np.asarray(action)[None])
        return m[0],float(v[0])


def main():
    r.Config.set_mode('fast')
    env=r.CrossingEnv(seed=7,obstacle_density=0)
    controller=LinearMPC(env)
    obstacle=env.obstacles[0]
    def cost(distance,margin):
        state=np.array([obstacle['x']-obstacle['r']-distance,obstacle['y'],0.,0.])
        # Isolate the nearest-distance map to test the scalar branches.
        old=env.get_min_dist
        env.get_min_dist=lambda x,y:distance
        try:return controller._obstacle_cost(state,margin)
        finally:env.get_min_dist=old
    for margin in [1.35,1.91,3.15,4.2,4.8]:
        influence=max(r.Config.OBSTACLE_INFLUENCE,margin)
        samples=np.linspace(-1,influence+1,1000)
        values=np.array([cost(d,margin) for d in samples])
        assert np.all(values>=0) and np.all(np.diff(values)<=1e-8)
        for junction in [margin,influence]:
            h=1e-6
            left=(cost(junction,margin)-cost(junction-h,margin))/h
            right=(cost(junction+h,margin)-cost(junction,margin))/h
            assert abs(left-right)<.03,(margin,junction,left,right)
    for mode in ['estimated']:
        fed=FedUMPCController(env,FixedPredictor(),uncertainty_mode=mode)
        state=np.array([-20.,-2.,3.,0.]);target=np.array([20.,-2.,3.,0.])
        candidates=np.asarray(fed._initial_guesses(state,target))
        batched=fed._cost_many(candidates,state,target)
        serial=np.asarray([fed._cost(c,state,target) for c in candidates])
        np.testing.assert_allclose(batched,serial,rtol=1e-6,atol=.01)
    with r.temporary_config(DEVICE=torch.device('cpu')):
        r.reset_rng(99);model=r.BNN()
        x=torch.zeros((3,6))
        r.reset_rng(1001);mean,var=model._mc_batch(x,num_samples=5)
        r.reset_rng(1001)
        with torch.no_grad():
            pred=model(x.unsqueeze(0).expand(5,3,6).reshape(15,6)).reshape(5,3,4)
        expected=(pred/model.delta_scale).var(0,unbiased=True).sum(1)
        np.testing.assert_allclose(var,expected.numpy(),rtol=1e-6)
        np.testing.assert_allclose(mean,pred.mean(0).numpy(),rtol=1e-6)
    class ConstantController:
        def get_action(self,state,target):return np.zeros(2)
    result=r.run_simulation([ConstantController() for _ in range(4)],env,seed=7)
    assert result['CR']>0
    np.testing.assert_allclose(result['SC']/100+result['VR'],1.)
    assert sum(v['violation_steps'] for v in result['RouteMetrics'])>=4*result['CR']
    for path in r.OUT.glob('*_runs.csv'):
        rows=list(csv.reader(path.open(newline='',encoding='utf-8')))
        assert all(len(row)==len(rows[0]) for row in rows),path
    print('PASS: C1 and monotone obstacle branches; scalar/batched score parity;')
    print('      dimensionless sample covariance; contact-inclusive compliance; CSV schemas.')


if __name__=='__main__':main()
