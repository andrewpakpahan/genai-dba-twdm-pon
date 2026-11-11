"""Packet-capable rotating BE visits within the bounded offline Limited round.

This isolated experiment leaves all historical engines unchanged. Every ONU
still receives a grant/REPORT visit. EF and already-reported AF retain their
Limited grants. Before optional AF prediction, a rotating subset of eligible
ONUs receives a BE reservation capable of one known maximum-size source packet.
Other BE is admitted as a complete eligible grant, never uniformly shrunk.

Only REPORT/history, applied prior grants, configured maximum packet size and
the persistent rotation state enter allocation. Current ONU queues and future
arrivals are not available to the OLT. The configured BE visit goal is best
effort when protected service and physical control overhead consume the budget.
"""
from importlib import import_module
import math
import numpy as np

bounded=import_module('20260913_0822_bounded_af_engine')
DECISION_DTYPE=np.dtype(bounded.DECISION_DTYPE.descr+[
    ('be_selected','?'),('be_eligible','?'),('be_reserved_bits','f8'),
    ('credit_cap_bits','f8'),('be_rotation_start','i2'),('be_rotation_next','i2'),
    ('be_selection_rank','i2'),('be_selection_budget_rejected','?'),
    ('be_optional_full_grant_deferred','?'),('be_reserved_visits_target','i2'),
    ('be_reserved_visits_actual','i2'),('max_packet_bits','f8'),
])
CYCLE_DTYPE=np.dtype(bounded.CYCLE_DTYPE.descr+[
    ('be_selected_visits','i4'),('be_eligible_visits','i4'),
    ('be_visits_requested','i4'),('be_quota_shortfall','i4'),('be_reserved_bits','f8'),
])


class RotatingBESimulator(bounded.BoundedSimulator):
    def __init__(self,config,mode='baseline',predictor=None,target_cycle_us=1200.,
                 credit_fraction=.5,activation_time=1.,window=50,
                 be_visits_per_cycle=16,max_packet_bits=12144.):
        super().__init__(config,mode,predictor,target_cycle_us,credit_fraction,activation_time,window)
        if (isinstance(be_visits_per_cycle,bool) or not isinstance(be_visits_per_cycle,int)
                or be_visits_per_cycle<0):
            raise ValueError('be_visits_per_cycle must be a nonnegative integer')
        if not np.isfinite(max_packet_bits) or max_packet_bits<=0:
            raise ValueError('max_packet_bits must be finite and positive')
        self.be_visits_per_cycle=be_visits_per_cycle
        self.max_packet_bits=float(max_packet_bits)
        self.rotation=np.zeros(config.channels,dtype=int)

    def decision_array(self):
        return np.concatenate(self.decision_rows) if self.decision_rows else np.empty(0,DECISION_DTYPE)

    def cycle_array(self):
        return np.asarray(self.cycle_rows,dtype=CYCLE_DTYPE)

    def _reserved_credit(self,baseline,rates,snapshots,reserved,credit_caps,ch,available,scale):
        cfg=self.config;group=cfg.onu_count//cfg.channels
        cursor=available+cfg.get_channel_offsets()[ch]
        credits=np.zeros(group);horizons=np.zeros(group)
        for j,u in enumerate(range(ch*group,(ch+1)*group)):
            cursor=math.ceil(cursor*1e9+.5)/1e9
            cursor+=baseline[u,0]/self.channel_rates[ch]
            horizons[j]=max(0.,cursor-snapshots[u]) if np.isfinite(snapshots[u]) else 0.
            credits[j]=scale*min(rates[u]*horizons[j],credit_caps[u])
            cursor+=(baseline[u,1]+credits[j])/self.channel_rates[ch]
            cursor+=reserved[u]/self.channel_rates[ch]
            report_end=cursor+cfg.report_tx_time
            cursor+=cfg.report_tx_time+cfg.guard_time
        return credits,horizons,report_end+cfg.report_propagation

    def _dba(self,now):
        cfg=self.config;n=cfg.onu_count;group=n//cfg.channels
        goal=min(self.be_visits_per_cycle,group)
        if self.mode=='baseline':
            super()._dba(now)
            old=self.decision_rows[-1];row=np.zeros(n,DECISION_DTYPE)
            for name in old.dtype.names:row[name]=old[name]
            row['be_selection_rank']=-1;row['max_packet_bits']=self.max_packet_bits
            self.decision_rows[-1]=row
            self.cycle_rows[-1]=tuple(self.cycle_rows[-1])+(0,0,0,0,0.)
            return
        if self.cycle==0:
            self.policy.reset();self.rotation[:]=0
            self.decision_rows=[];self.cycle_rows=[]
        cycle=self.cycle+1
        reports=self.reports.copy();previous=self.previous_grants.copy()
        rates=self.policy.observe(reports,previous,self.intervals.copy(),
            self.report_receive_times.copy(),now,cycle)
        base=self._baseline(reports,now,previous,self.intervals,cycle)
        snapshots=self.report_receive_times-cfg.report_tx_time-cfg.report_propagation
        available=float(np.max(self.next_available))
        if available<now:available=now+cfg.gate_tx_time
        available+=cfg.gate_tx_time
        deadline=now+self.target_cycle_us*1e-6
        grants=base.copy();grants[:,2]=0.
        reserved=np.zeros(n);selected=np.zeros(n,bool)
        rank=np.full(n,-1,dtype=int);budget_rejected=np.zeros(n,bool)
        extra_deferred=np.zeros(n,bool)
        # A small reported queue can be served completely without padding it to
        # the maximum packet size. Partial small queues are not guessed at.
        minimum=np.minimum(reports[:,2],self.max_packet_bits)
        eligible=(minimum>0)&(base[:,2]>=minimum-1e-7)
        credits=np.zeros(n);horizons=np.zeros(n)
        scales=np.zeros(cfg.channels);infeasible=np.zeros(cfg.channels,bool)
        rotation_start=self.rotation.copy();selected_counts=np.zeros(cfg.channels,dtype=int)
        credit_caps=np.minimum(self.credit_fraction*base[:,2],base[:,2])
        for ch in range(cfg.channels):
            lo_u=ch*group;sl=slice(lo_u,lo_u+group)
            order=lo_u+(self.rotation[ch]+np.arange(group))%group
            infeasible[ch]=self._channel_schedule(grants,ch,available)[0]>deadline+1e-12
            last_selected=None
            if not infeasible[ch]:
                for u in order:
                    if selected_counts[ch]>=goal:break
                    if not eligible[u]:continue
                    grants[u,2]=minimum[u]
                    if self._channel_schedule(grants,ch,available)[0]<=deadline:
                        reserved[u]=minimum[u];selected[u]=True
                        rank[u]=selected_counts[ch];selected_counts[ch]+=1
                        last_selected=int(u-lo_u)
                    else:
                        grants[u,2]=0.;budget_rejected[u]=True
                credit_caps[sl]=np.minimum(self.credit_fraction*base[sl,2],
                                            np.maximum(0.,base[sl,2]-reserved[sl]))
                c,h,finish=self._reserved_credit(base,rates,snapshots,reserved,
                    credit_caps,ch,available,1.)
                scale=1.
                if finish>deadline:
                    low,high=0.,1.
                    for _ in range(30):
                        mid=(low+high)/2
                        if self._reserved_credit(base,rates,snapshots,reserved,
                                credit_caps,ch,available,mid)[2]<=deadline:low=mid
                        else:high=mid
                    scale=low
                    c,h,_=self._reserved_credit(base,rates,snapshots,reserved,
                        credit_caps,ch,available,scale)
                credits[sl]=c;horizons[sl]=h;scales[ch]=scale
                grants[sl,1]+=c
                # Use complete remaining eligible BE grants where they fit.
                # Selected reservations survive if enlarging their grant fails.
                for u in order:
                    candidate=max(0.,base[u,2]-credits[u])
                    if not eligible[u] or candidate<minimum[u]-1e-7:continue
                    if candidate<=grants[u,2]+1e-7:continue
                    saved=grants[u,2];grants[u,2]=candidate
                    if self._channel_schedule(grants,ch,available)[0]>deadline:
                        grants[u,2]=saved;extra_deferred[u]=True
            else:
                _,horizons[sl],_=self._reserved_credit(base,rates,snapshots,reserved,
                    credit_caps,ch,available,0.)
            self.rotation[ch]=(last_selected+1)%group if last_selected is not None else (self.rotation[ch]+1)%group
        caps=np.asarray(cfg.get_base_caps())
        positive=grants[:,2]>1e-7
        if (not np.isfinite(grants).all() or (grants<0).any()
                or not np.array_equal(grants[:,0],base[:,0])
                or (grants[:,1]<base[:,1]-1e-7).any()
                or (grants[:,1]>base[:,1]+credits+1e-7).any()
                or (credits+grants[:,2]>base[:,2]+1e-7).any()
                or (grants.sum(axis=1)>caps+1e-7).any()
                or (grants[selected,2]<reserved[selected]-1e-7).any()
                or (grants[positive,2]<minimum[positive]-1e-7).any()):
            raise AssertionError('Rotating BE packet capability or protected Limited grant violated')
        starts=np.zeros((n,4));lengths=np.zeros_like(starts)
        next_receives=np.zeros(cfg.channels)
        wavelengths=np.arange(n)//group
        for ch in range(cfg.channels):
            sl=slice(ch*group,(ch+1)*group)
            next_receives[ch],cursor,starts[sl],lengths[sl]=self._channel_schedule(grants,ch,available,output=True)
            self.next_available[ch]=cursor
        self.cycle=cycle;self.previous_grants[:]=grants
        self.dba_times.append((now,available))
        self._dispatch(now,starts,lengths,wavelengths,cycle)
        verified=not cfg.shared_upstream_serialization or cfg.channels==1
        row=np.zeros(n,DECISION_DTYPE)
        row['cycle']=cycle;row['onu']=np.arange(n);row['wavelength']=wavelengths
        row['decision_time']=now;row['report_snapshot_time']=snapshots
        row['report_receive_time']=self.report_receive_times;row['report_interval_s']=self.intervals
        row['window_start_time']=self.policy.times[0];row['window_end_time']=self.policy.times[-1]
        row['window_count']=len(self.policy.features);row['prediction_active']=self.policy.active
        row['reports']=reports;row['previous_grants']=previous
        row['baseline']=base;row['final']=grants;row['cap_bits']=caps
        row['predicted_af_rate_bps']=rates;row['credit_horizon_s']=horizons
        row['actual_af_horizon_s']=np.maximum(0.,starts[:,1]-snapshots)
        row['af_credit_bits']=credits;row['be_deferred_bits']=base[:,2]-grants[:,2]-credits
        row['credit_scale']=scales[wavelengths];row['target_cycle_us']=self.target_cycle_us
        row['planned_interval_us']=(next_receives[wavelengths]-now)*1e6
        row['target_infeasible']=infeasible[wavelengths];row['timing_bound_verified']=verified
        row['be_selected']=selected;row['be_eligible']=eligible;row['be_reserved_bits']=reserved
        row['credit_cap_bits']=credit_caps
        row['be_rotation_start']=rotation_start[wavelengths];row['be_rotation_next']=self.rotation[wavelengths]
        row['be_selection_rank']=rank;row['be_selection_budget_rejected']=budget_rejected
        row['be_optional_full_grant_deferred']=extra_deferred
        row['be_reserved_visits_target']=goal;row['be_reserved_visits_actual']=selected_counts[wavelengths]
        row['max_packet_bits']=self.max_packet_bits
        self.decision_rows.append(row)
        selected_total=int(selected.sum());requested=goal*cfg.channels
        self.cycle_rows.append((cycle,now,float(next_receives.max()),
            float((next_receives.max()-now)*1e6),self.target_cycle_us,bool(infeasible.any()),
            verified,float(credits.sum()),float(row['be_deferred_bits'].sum()),
            selected_total,int(eligible.sum()),requested,requested-selected_total,float(reserved.sum())))


RotatingBESimulation=RotatingBESimulator
