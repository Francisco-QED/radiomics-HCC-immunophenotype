"""Peritumoral annulus masks (run inside 3D Slicer, headless):
  /Applications/Slicer.app/Contents/MacOS/Slicer --no-main-window --python-script peritumor_annulus_slicer.py --exit-after-startup
Annulus = outer rim (in-plane distance <= PERI_R_OUT mm outside the tumor edge, clipped to the liver mask) + inner rim (<= PERI_R_IN mm inside the edge, not clipped), computed slice by slice on a 0.5 mm refined grid
from the Segment_1 tumor ROI on the image grid. Reproduces the project annulus masks voxel for voxel (validated on SA1 5/2 masks).
Environment: PERI_OUT (output folder, required) | PERI_R_OUT (default 5) | PERI_R_IN (default 2) | PERI_COHORTS (default 'SA1,SA2'; may include SA2_post) | PERI_READER (Reader1/Reader2: expand that reader's ROIs on the SA1 images)
             | PERI_ONLY (comma list of cases, e.g. P001_pvp) | PERI_COPY_LIVER (1 = also copy the liver masks into the output)
             | PERI_SEG_ROOT (tumor-segmentation root) | PERI_CLEAN_ROOT (images + liver-mask root)
Inputs (read only): tumor ROIs in Total_segmentations on the T7; images + liver masks in '/Volumes/LaCie2/clean segs 10-23-26'.
"""
import slicer, glob, os, csv, shutil, time, traceback, sys, numpy as np
from scipy import ndimage as ndi
S=os.environ.get("PERI_SEG_ROOT","/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations")   # READ ONLY
D=os.environ.get("PERI_CLEAN_ROOT","/Volumes/LaCie2/clean segs 10-23-26")                                            # images + liver (READ ONLY)
O=os.environ.get("PERI_OUT")           # all outputs
assert O, "PERI_OUT is required"
ONLY=[c for c in os.environ.get("PERI_ONLY","").split(",") if c]
assert not os.path.abspath(O).startswith(S) and not os.path.abspath(O).startswith(D)
R_OUT=float(os.environ.get("PERI_R_OUT","5")); R_IN=float(os.environ.get("PERI_R_IN","2")); FINE=0.5; COHORTS=[c for c in os.environ.get("PERI_COHORTS","SA1,SA2").split(",") if c]; READER=os.environ.get("PERI_READER"); COPY_LIVER=os.environ.get("PERI_COPY_LIVER")=="1"
LOG=O+("/qc_peritumor_%s.csv" % (READER or "_".join(COHORTS))); ERR=O+("/errors_%s.log" % (READER or "_".join(COHORTS)))
os.makedirs(O,exist_ok=True)
def odd_factor(s):
    f=int(np.ceil(s/FINE)); return f if f%2 else f+1
def inner_dist(T,sp):
    sp=sp.copy(); sp[0]=1e6                      # in-plane only
    idx=np.argwhere(T); lo=idx.min(0); hi=idx.max(0)+1
    pad=np.ceil(max(R_OUT,R_IN)/sp).astype(int)+2
    lo=np.maximum(lo-pad,0); hi=np.minimum(hi+pad,T.shape)
    sl=tuple(slice(a,b) for a,b in zip(lo,hi)); Tc=T[sl]
    f=np.array([1]+[odd_factor(s) for s in sp[1:]]); fsp=sp/f
    Tf=np.repeat(np.repeat(np.repeat(Tc,f[0],0),f[1],1),f[2],2)
    h=fsp[1:].min()/2
    di=np.maximum(ndi.distance_transform_edt(Tf,sampling=fsp)-h,0)
    c=tuple(slice(k//2,None,k) for k in f)
    d_in=np.zeros(T.shape); d_in[sl]=di[c]
    return d_in
def save_seg(arr,name,color,img,path):
    o=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    o.SetReferenceImageGeometryParameterFromVolumeNode(img)
    sid=o.GetSegmentation().AddEmptySegment("",name,color)
    slicer.util.updateSegmentBinaryLabelmapFromArray(arr.astype(np.uint8),o,sid,img)
    assert slicer.util.saveNode(o,path), path
    slicer.mrmlScene.RemoveNode(o)
cols=["case","status","note","n_tumor_segs_used","tumor_ml","tumor_in_liver_frac","inner_ml","outer_ml","annulus_ml","annulus_nslices",
      "inner_tumor_frac","inner_is_entire_tumor","annulus_tumor_overlap_voxels"]
done=set()
if os.path.exists(LOG): done={r["case"] for r in csv.DictReader(open(LOG)) if r["status"]!="ERROR"}
newfile=not os.path.exists(LOG)
fh=open(LOG,"a",newline=""); w=csv.DictWriter(fh,fieldnames=cols)
if newfile: w.writeheader()
cases=[]
if READER:
    for pid in sorted(os.listdir(f"{S}/Inter-reader/{READER}")):
        if pid.startswith("P"):
            for seq in ("adc","ap","dwi_b1","dwi_b2","hbp","pre","pvp","t2","tp"): cases.append((READER,f"{D}/SA1/{pid}",f"{pid}_{seq}"))
else:
    for sa in COHORTS:
        for p in sorted(glob.glob(f"{D}/{sa}/P*/*.nrrd")):
            b=os.path.basename(p)
            if b.startswith("._") or b.endswith(".seg.nrrd"): continue
            cases.append((sa,os.path.dirname(p),b[:-5]))
if ONLY: cases=[c for c in cases if c[2] in ONLY]
t0=time.time()
for n,(sa,pdir,base) in enumerate(cases):
    pid=base.split("_")[0]; key=f"{sa}/{pid}/{base}"
    if key in done: continue
    row={"case":key}
    try:
        tf=(f"{S}/Inter-reader/{READER}/{pid}/{base}.seg.nrrd" if READER else f"{S}/{sa.replace('SA2_post','SA2-post')}/{pid}/{base}.seg.nrrd")
        if not os.path.exists(tf):
            row.update(status="SKIPPED",note="no tumor segmentation"); w.writerow(row); fh.flush(); continue
        img=slicer.util.loadVolume(f"{pdir}/{base}.nrrd")
        lv=slicer.util.loadLabelVolume(f"{pdir}/{base}_liver.seg.nrrd")
        L=slicer.util.arrayFromVolume(lv)>0
        sn=slicer.util.loadSegmentation(tf); seg=sn.GetSegmentation()
        use=[seg.GetNthSegmentID(i) for i in range(seg.GetNumberOfSegments()) if seg.GetNthSegment(i).GetName()=="Segment_1"]
        if not use:
            row.update(status="SKIPPED",note="no Segment_1"); w.writerow(row); fh.flush(); slicer.mrmlScene.Clear(0); continue
        T=np.zeros(L.shape,bool)
        for sid in use: T|=slicer.util.arrayFromSegmentBinaryLabelmap(sn,sid,img)>0
        if not T.any():
            row.update(status="SKIPPED",note="tumor empty on image grid"); w.writerow(row); fh.flush(); slicer.mrmlScene.Clear(0); continue
        sp=np.array(img.GetSpacing())[::-1]; vox=np.prod(sp)/1000
        d_out=ndi.distance_transform_edt(~T,sampling=np.array([1e6,sp[1],sp[2]]))  # centre-to-centre, in-plane
        outer=((d_out<=R_OUT)&~T&L) if R_OUT>0 else np.zeros_like(T,bool)             # clipped to liver; strict zero means no outer component
        inner=(T&(inner_dist(T,sp)<=R_IN)) if R_IN>0 else np.zeros_like(T,bool)       # strict zero means no inner component
        ann=outer|inner
        od=f"{O}/{sa}/{pid}"; os.makedirs(od,exist_ok=True)
        c1=seg.GetSegment(use[0]).GetColor()
        save_seg(T,"Tumor",c1,img,f"{od}/{base}_tumor.seg.nrrd")
        save_seg(ann,f"Annulus_{R_OUT:g}out_{R_IN:g}in",(0.9,0.3,0.2),img,f"{od}/{base}_annulus.seg.nrrd")
        if COPY_LIVER: shutil.copy2(f"{pdir}/{base}_liver.seg.nrrd",f"{od}/{base}_liver.seg.nrrd")
        notes=[]
        if len(use)>1: notes.append("multiple Segment_1 merged")
        if seg.GetNumberOfSegments()>len(use): notes.append("other segments ignored")
        fr=(T&L).sum()/T.sum()
        if fr<0.9: notes.append("tumor <90% inside liver mask")
        inner_frac=float(inner.sum()/T.sum())
        row.update(status="OK",note="; ".join(notes),n_tumor_segs_used=len(use),tumor_ml=round(T.sum()*vox,2),tumor_in_liver_frac=round(fr,3),
                   inner_ml=round(inner.sum()*vox,2),outer_ml=round(outer.sum()*vox,2),annulus_ml=round(ann.sum()*vox,2),annulus_nslices=int(ann.any(axis=(1,2)).sum()),
                   inner_tumor_frac=round(inner_frac,4),inner_is_entire_tumor=int(np.array_equal(inner,T)),annulus_tumor_overlap_voxels=int((ann&T).sum()))
        w.writerow(row); fh.flush()
    except Exception as e:
        open(ERR,"a").write(f"{key}\n{traceback.format_exc()}\n")
        row.update(status="ERROR",note=str(e)[:150]); w.writerow(row); fh.flush()
    slicer.mrmlScene.Clear(0)
    if n%25==0: print(f"PROGRESS {n+1}/{len(cases)} elapsed {time.time()-t0:.0f}s",flush=True)
print("ALLDONE",flush=True)
