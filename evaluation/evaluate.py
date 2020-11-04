import torch
from torch.utils.data import DataLoader
from lib.plotting import plot_recall_curves
from lib.utils import compute_angular_error, compute_rotation_errors, compute_translation_errors
from lib.tensorlist import TensorListList
from lib.visualization import save_point_clouds_open3d

def evaluate_registration(method, batch, vis=None, save=False):
    data, info = batch
    out = method(data)
    R_err = compute_rotation_errors(out["Rs"].cpu(), info['R_gt'])
    t_err = compute_translation_errors(out["ts"].cpu(), info['t_gt'], out["Rs"].cpu(), info['R_gt'])
    if not vis is None:
        vis(out, info, data)
    if save:
        save_point_clouds_open3d(out, info)
    return dict(R_err=R_err, t_err=t_err, time=out["time"])

def benchmark(methods, dataset, job_name, batch_size=1, plot=True, vis=None, num_workers=0,
              collate_fn=None, epoch=0, max_err_R=8.0, max_err_t=2.0, success_R=4, success_t=0.3,
              save_errors=False, save=False):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    errors = dict()
    for m in methods:
        m.eval()
        errors[m.params.name] = []

    cnt = 0
    names = []
    for batch in loader:
        print("evaluate batch %d ..." % (cnt))
        data, info = batch
        names.extend(info["names"])
        for m in methods:
            print(m.params.name)
            errors[m.params.name].append(evaluate_registration(m, batch, vis, save=save))

        cnt = cnt + 1

    eval_info = dict()
    for m in methods:
        if isinstance(errors[m.params.name][0]["R_err"], TensorListList):
            R_err = torch.cat([torch.cat(e["R_err"].flatten_out().list(), dim=0) for e in errors[m.params.name]], dim=0)
            t_err = torch.cat([torch.cat(e["t_err"].flatten_out().list(), dim=0) for e in errors[m.params.name]], dim=0)
        elif isinstance(errors[m.params.name][0]["R_err"], torch.tensor):
            R_err = torch.cat([e["R_err"] for e in errors[m.params.name]], dim=0)
            t_err = torch.cat([e["t_err"] for e in errors[m.params.name]], dim=0)
        else:
            raise NotImplementedError

        tot_time = sum([e["time"] for e in errors[m.params.name]])

        ang_err = compute_angular_error(R_err)

        N = float(ang_err.numpy().size)
        M = float((ang_err < success_R).sum())
        print(m.params.name + " " + str(success_R) + " degree recall: ", M / N)

        Nt = float(t_err.numpy().size)
        Mt = float((t_err < success_t).sum())
        print(m.params.name + " " + str(success_t) + " meters translation error recall: ", Mt / Nt)

        M_all = float(((t_err < success_t) * (ang_err < success_R)).sum())
        print(m.params.name + " total error recall: ", M_all / N)

        mask = (t_err < success_t) * (ang_err < success_R)
        t_merr = t_err[mask].mean().item()
        ang_merr = ang_err[mask].mean().item()

        print(m.params.name + " rot success mean error: ", ang_merr)
        print(m.params.name + " t success mean error: ", t_merr)
        print(m.params.name + " mean time: ", tot_time/N)

        if save_errors:
            eval_info[m.params.name] = {"R_recall": M / N, "t_recall": Mt / Nt, "total_recall": M_all / N,
                                    "R_err": ang_err, "t_err": t_err, "ang_merr": ang_merr, "t_merr": t_merr,
                                        "tot_time": tot_time}
        else:
            eval_info[m.params.name] = {"R_recall": M / N, "t_recall": Mt / Nt, "total_recall": M_all / N,
                                        "ang_merr": ang_merr, "t_merr": t_merr, "tot_time": tot_time}

        if "reg_time" in errors[m.params.name][0]:
            reg_time = sum([e["reg_time"] for e in errors[m.params.name]])
            eval_info[m.params.name]["reg_time"] = reg_time
            print(m.params.name + " mean reg time: ", reg_time/N)

    if plot:
        plot_recall_curves(errors, job_name + "/R_err" + ("eval_plot_%d.png" % epoch), key="R_err",
                           max_err=max_err_R, unit='°')
        plot_recall_curves(errors, job_name + "/t_err" + ("eval_plot_%d.png" % epoch), key="t_err",
                           max_err=max_err_t, unit='cm')

    eval_info["names"] = names
    return eval_info