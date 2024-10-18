clc
clear


[r, res] = mosekopt('read(probmosek.ptf)');

%param.MSK_IPAR_LOG_FEAS_REPAIR = 3;
%res.prob.primalrepair = [];
%param.MSK_IPAR_INFEAS_REPORT_AUTO = 'MSK_ON';
%param.MSK_IPAR_INFEAS_REPORT_LEVEL = 2;
%param.MSK_IPAR_PRESOLVE_USE = 'MSK_PRESOLVE_MODE_OFF';


[r,res]=mosekopt('minimize',res.prob);