
#include "TFile.h"
#include "TString.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH2.h"
#include "TTree.h"
#include <iostream>
#include "TEfficiency.h"
#include "TStyle.h"
#include "../interface/helpers.h"

#include "../interface/globals.h"






int plotscript(int argc, char* argv[]){


    global::setTrees();
    TCanvas *cv=createCanvas();
    legends::buildLegends();
    /////////standard legends

    gStyle->SetOptStat(0);

    compareEfficiency eff_energy1_5("true_e", "is_true && n_true <= 5", "(is_true && is_reco && n_true <= 5)",5,0,200,"p(t) [GeV]","efficiency");
    compareEfficiency eff_energy5_10("true_e", "is_true && n_true <= 10 && n_true>5", "(is_true && is_reco && n_true <= 10 && n_true>5)",5,0,200,"Momentum [GeV]","Efficiency");
    std::vector<compareEfficiency*> eff_mom = {&eff_energy1_5, &eff_energy5_10};



    eff_energy1_5.DrawAxes();
    eff_energy1_5.AxisHisto()->GetYaxis()->SetRangeUser(0.85,1.01);

    for(int i=0;i<eff_mom.size();i++){
        eff_mom.at(i)->setOCLineColourAndStyle(-1,2-i);
        eff_mom.at(i)->setClassicLineColourAndStyle(-1,2-i);
        eff_mom.at(i)->Draw("same,P");
    }


    legends::legend_smaller->Draw("same");
    cv->Print("mom_efficiency.pdf");



    ///////////////////////////////////////////

    compareEfficiency eff_n_true("n_true", "is_true", "(is_true && is_reco)",15,0.75,25.5,"Particles per event","Efficiency");
    eff_n_true.DrawAxes();
    eff_n_true.AxisHisto()->GetYaxis()->SetRangeUser(0.65,1.01);
    eff_n_true.Draw("same,P");


    legends::legend_simplest->Draw("same");

    cv->Print("N_efficiency.pdf");

    ///////////////////////////////////////////


    compareEfficiency fr_energy1_5("reco_e", "(is_reco && n_true <= 5)", "!is_true && is_reco && n_true <= 5",15,0,200,"p(r) [GeV]","fake rate");
    compareEfficiency fr_energy5_10("reco_e", "(is_reco && n_true <= 10 && n_true > 5)", "!is_true && is_reco && n_true <= 10 && n_true > 5",15,0,200,"p(reco) [GeV]","Fake rate");
    compareEfficiency fr_energy10_15("reco_e", "(is_reco && n_true <= 15 && n_true > 10)", "!is_true && is_reco && n_true <= 15 && n_true > 10",15,0,200,"p(reco) [GeV]","Fake rate");


    fr_energy1_5.setOCLineColourAndStyle(-1,3);
    fr_energy1_5.setClassicLineColourAndStyle(-1,3);
    fr_energy5_10.setOCLineColourAndStyle(-1,2);
    fr_energy5_10.setClassicLineColourAndStyle(-1,2);

    fr_energy1_5.DrawAxes();
    fr_energy1_5.AxisHisto()->GetYaxis()->SetRangeUser(1e-3,4.);
    fr_energy1_5.Draw("same","");
    fr_energy5_10.Draw("same","");
    fr_energy10_15.Draw("same","");

    placeLegend(legends::legend_full, 0.5, 0.55)->Draw("same");

    cv->SetLogy();
    cv->Print("fake_rate.pdf");
    cv->SetLogy(0);
    /////


    compareTH1D resolution1_5("reco_e/true_e","is_true && is_reco && n_true <= 5",51,0.8,1.2,"p(r)/p(t)","A.U.");
    compareTH1D resolution5_10("reco_e/true_e","is_true && is_reco && n_true <= 10 && n_true>5",51,0.8,1.2,"p(r)/p(t)","# particles");
    compareTH1D resolution10_15("reco_e/true_e","is_true && is_reco && n_true <= 15 && n_true>10",51,0.8,1.2,"Momentum resolution","# particles");

    resolution1_5.DrawAxes();
    //resolution1_5.AxisHisto()->GetYaxis()->SetRangeUser(0,18000);
    resolution1_5.setOCLineColourAndStyle(-1,3);
    resolution1_5.setClassicLineColourAndStyle(-1,3);
    resolution5_10.setOCLineColourAndStyle(-1,2);
    resolution5_10.setClassicLineColourAndStyle(-1,2);
    resolution1_5.Draw("same,HIST","PF");
    resolution5_10.Draw("same,HIST","PF");
    resolution10_15.Draw("same,HIST","PF");


    legends::legend_onlyn->Draw("same");

    cv->Print("resolution_pf.pdf");

    resolution1_5.DrawAxes();
    resolution1_5.Draw("same,HIST","OC");
    resolution5_10.Draw("same,HIST","OC");
    resolution10_15.Draw("same,HIST","OC");
    legends::legend_onlyn->Draw("same");

    cv->Print("resolution_oc.pdf");

    ///////////////////////////////////////////


    compareProfile variance1_5("(reco_e/true_e - 1)**2:true_e",  "abs(reco_e/true_e - 1)<0.1 && is_true && is_reco && n_true <= 5",10,0,200,"p(t) [GeV]","variance of (p(r)/p(t) - 1.)");
    compareProfile variance5_10("(reco_e/true_e - 1)**2:true_e", "abs(reco_e/true_e - 1)<0.1 &&is_true && is_reco && n_true <= 10 && n_true>5",10,0,200,"True momentum [GeV]","Variance");
    compareProfile variance10_15("(reco_e/true_e - 1)**2:true_e","abs(reco_e/true_e - 1)<0.1 &&is_true && is_reco && n_true <= 15 && n_true>10",10,0,200,"True momentum [GeV]","Variance");

    variance1_5.setOCLineColourAndStyle(-1,3);
    variance1_5.setClassicLineColourAndStyle(-1,3);
    variance5_10.setOCLineColourAndStyle(-1,2);
    variance5_10.setClassicLineColourAndStyle(-1,2);

    variance1_5.DrawAxes();
    variance1_5.AxisHisto()->GetYaxis()->SetRangeUser(0.,0.003);
    variance1_5.Draw("same","PF");
    variance5_10.Draw("same","PF");
    variance10_15.Draw("same","PF");

    cv->Print("variance_pf.pdf");

    variance1_5.DrawAxes();
    variance1_5.AxisHisto()->GetYaxis()->SetRangeUser(0.,0.003);
    variance1_5.Draw("same","OC");
    variance5_10.Draw("same","OC");
    variance10_15.Draw("same","OC");

    cv->Print("variance_oc.pdf");


    variance1_5.DrawAxes();
    variance1_5.Draw("same","");
    variance5_10.Draw("same","");
    variance10_15.Draw("same","");

    placeLegend(legends::legend_full, 0.2, 0.55);
    legends::legend_full->Draw("same");

    cv->Print("variance.pdf");

    compareProfile elec_variance1_5("(reco_e/true_e - 1)**2:true_e",  "true_id && abs(reco_e/true_e - 1)<0.1 && is_true && is_reco && n_true <= 5",10,0,200,"p(t) [GeV]","variance of (p(r)/p(t) - 1.)");
    compareProfile elec_variance5_10("(reco_e/true_e - 1)**2:true_e", "true_id && abs(reco_e/true_e - 1)<0.1 &&is_true && is_reco && n_true <= 10 && n_true>5",10,0,200,"True momentum [GeV]","Variance");
    compareProfile elec_variance10_15("(reco_e/true_e - 1)**2:true_e","true_id && abs(reco_e/true_e - 1)<0.1 &&is_true && is_reco && n_true <= 15 && n_true>10",10,0,200,"True momentum [GeV]","Variance");

    elec_variance1_5.setOCLineColourAndStyle(-1,3);
    elec_variance1_5.setClassicLineColourAndStyle(-1,3);
    elec_variance5_10.setOCLineColourAndStyle(-1,2);
    elec_variance5_10.setClassicLineColourAndStyle(-1,2);

    elec_variance1_5.DrawAxes();
    elec_variance1_5.AxisHisto()->GetYaxis()->SetRangeUser(0.,0.003);
    elec_variance1_5.Draw("same","");
    elec_variance5_10.Draw("same","");
    elec_variance10_15.Draw("same","");

    placeLegend(legends::legend_full, 0.2, 0.55);
    legends::legend_full->Draw("same");

    cv->Print("elec_variance.pdf");

    compareProfile gamma_variance1_5("(reco_e/true_e - 1)**2:true_e",  "!true_id && abs(reco_e/true_e - 1)<0.1 && is_true && is_reco && n_true <= 5",10,0,200,"p(t) [GeV]","variance of (p(r)/p(t) - 1.)");
    compareProfile gamma_variance5_10("(reco_e/true_e - 1)**2:true_e", "!true_id && abs(reco_e/true_e - 1)<0.1 &&is_true && is_reco && n_true <= 10 && n_true>5",10,0,200,"True momentum [GeV]","Variance");
    compareProfile gamma_variance10_15("(reco_e/true_e - 1)**2:true_e","!true_id && abs(reco_e/true_e - 1)<0.1 &&is_true && is_reco && n_true <= 15 && n_true>10",10,0,200,"True momentum [GeV]","Variance");

    gamma_variance1_5.setOCLineColourAndStyle(-1,3);
    gamma_variance1_5.setClassicLineColourAndStyle(-1,3);
    gamma_variance5_10.setOCLineColourAndStyle(-1,2);
    gamma_variance5_10.setClassicLineColourAndStyle(-1,2);

    gamma_variance1_5.DrawAxes();
    gamma_variance1_5.AxisHisto()->GetYaxis()->SetRangeUser(0.,0.006);
    gamma_variance1_5.Draw("same","");
    gamma_variance5_10.Draw("same","");
    gamma_variance10_15.Draw("same","");

    placeLegend(legends::legend_full, 0.6, 0.55);
    legends::legend_full->Draw("same");

    cv->Print("gamma_variance.pdf");


    compareTH1D outside_var1_5("true_e","1./1000.*(abs(reco_e/true_e - 1)>0.1 && is_true && is_reco && n_true <= 5)",10,0,200,"True momentum [GeV]","mis-reco. fraction [%]",false);
    compareTH1D outside_var5_10("true_e","1./1000.*(abs(reco_e/true_e - 1)>0.1 &&is_true && is_reco && n_true <= 10 && n_true>5)",10,0,200,"True momentum [GeV]","Misidentified fraction [%]",false);
    compareTH1D outside_var10_15("true_e","1./1000.*(abs(reco_e/true_e - 1)>0.1 &&is_true && is_reco && n_true <= 15 && n_true>10)",10,0,200,"True momentum [GeV]","Misidentified fraction [%]",false);

    outside_var1_5.setOCLineColourAndStyle(-1,3);
    outside_var1_5.setClassicLineColourAndStyle(-1,3);
    outside_var5_10.setOCLineColourAndStyle(-1,2);
    outside_var5_10.setClassicLineColourAndStyle(-1,2);

    outside_var1_5.DrawAxes();
    outside_var1_5.AxisHisto()->GetYaxis()->SetRangeUser(0.,2.6);
    outside_var1_5.Draw("same","");
    outside_var5_10.Draw("same","");
    outside_var10_15.Draw("same","");

    placeLegend(legends::legend_full, 0.6, 0.55);
    legends::legend_full->Draw("same");

    cv->Print("misidentified.pdf");

    return 0;
}


int main(int argc, char* argv[]){
    return plotscript(argc,argv);
}
