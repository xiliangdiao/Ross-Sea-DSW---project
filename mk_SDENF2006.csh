#!/bin/csh -f

#set echo
#set verbose

set yearb  = 2100
set yeare  = 2101

setenv YEARB $yearb
setenv YEARE $yeare
set fyearb = `printf "%04d" $yearb`
set fyeare = `printf "%04d" $yeare`

# foreach case ( b.e12.B2000.f19_g16.ea_id_aero.002 b.e12.B2000.f19_g16.ea_id_aero.003 b.e12.B2000.f19_g16.ea_id_aero.004 b.e12.B2000.f19_g16.ea_id_aero.005 b.e12.B2000.f19_g16.ea_id_aero.006 b.e12.B2000.f19_g16.ea_id_aero.007 b.e12.B2000.f19_g16.ea_id_aero.008 b.e12.B2000.f19_g16.ea_id_aero.009 b.e12.B2000.f19_g16.ea_id_aero.010 )
#  foreach case ( b.e12.B2000.f19_g16.ea_id_aero b.e12.B2000.f19_g16.ctrl )
 foreach case (B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF)

  # set inpath = /data3/lix/model/${case}/ocn/hist
  #set inpath = /data3/model/cesm/test_cesm1_1_2/archive/${case}/ocn/hist
  set inpath = /home/hires_pi_ctrl/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF/ocn/monthly
  set inpath2 = /home/hires_pi_ctrl/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF/ocn/daily
  set rawpath  = /home/luoyy/DXL/TRY/lala/POP_WMF-master
  set workpath = /home/luoyy/DXL/TRY/lala/work5

 #if !(-d ${workpath}) mkdir -p ${workpath}

#rm -rf ${workpath}/*
cd ${workpath}
#ln ${rawpath}/pop_sdenf.singlefile.py .
ln ${rawpath}/parameter_template.nc .

set year = $yearb
  while ($year <= $yeare)
  set fyear = `printf "%04d" ${year}`
  set mon = 1
    while ($mon <= 12)
    set fmon = `printf "%02d" $mon`
    ln -sf ${inpath}/${case}.pop.h.${fyear}-${fmon}.nc .
    
    ln -sf ${inpath2}/${case}.pop.h.nday1.${fyear}-${fmon}-01.nc .

    python3 pop_sdenf.singlefile.py -s ${case}.pop.h.${fyear}-${fmon}.nc
   # mv ${case}.pop.h.${fyear}-${fmon}.SDEN_F.nc /home/luoyy/DXL/TRY/lala/work1

    @ mon++
    end
  @ year++
  end
end

#
#ncrcat ${case}.pop.h.????-??.

