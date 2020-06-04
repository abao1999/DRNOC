
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



int main(){

    global::setTrees();
    TCanvas *cv=createCanvas();
    gStyle->SetOptStat(0);
    legends::buildLegends();

    std::vector<TString> pus={"pu0.2", "pu0.0","pu0.5","pu0.8"};

    int resobins=15;
    double resobinhigh=200;

    compareJetMassRes jetmassres_pu0 (resobins,0,resobinhigh, "m_{j}(t) [GeV]","<m_{j}(r)/m_{j}(t)>","pu0.0",true);
    compareJetMassRes jetmassres_pu02(resobins,0,resobinhigh, "m_{j}(t) [GeV]","#sigma(m_{j}(r)","pu0.2",true);
    compareJetMassRes jetmassres_pu08(resobins,0,resobinhigh, "m_{j}(t) [GeV]","#sigma(m_{j}(r)","pu0.8",true);


    jetmassres_pu0.setClassicLineColourAndStyle(-1,3);
    jetmassres_pu02.setClassicLineColourAndStyle(-1,2);
    jetmassres_pu0.setOCLineColourAndStyle(-1,3);
    jetmassres_pu02.setOCLineColourAndStyle(-1,2);

    jetmassres_pu0.DrawAxes();
    jetmassres_pu0.AxisHisto()->GetYaxis()->SetRangeUser(0.91, 1.39);
    jetmassres_pu0.Draw( "same,HIST");
    jetmassres_pu02.Draw("same,HIST");
    jetmassres_pu08.Draw("same,HIST");



    placeLegend(legends::legend_pu_00_02_08, 0.6, 0.5)->Draw("same");

    cv->Print("jetmass_res_offset.pdf");


    compareJetMassRes jetmassres_var_pu0 (resobins,0,resobinhigh, "m_{j}(t) [GeV]","#sigma m_{j}(r)/<m_{j}(r)>","pu0.0",false);
    compareJetMassRes jetmassres_var_pu02(resobins,0,resobinhigh, "m_{j}(t) [GeV]","#sigma(m_{j}(r)","pu0.2",false);
    compareJetMassRes jetmassres_var_pu08(resobins,0,resobinhigh, "m_{j}(t) [GeV]","#sigma(m_{j}(r)","pu0.8",false);


    jetmassres_var_pu0.setClassicLineColourAndStyle(-1,3);
    jetmassres_var_pu02.setClassicLineColourAndStyle(-1,2);
    jetmassres_var_pu0.setOCLineColourAndStyle(-1,3);
    jetmassres_var_pu02.setOCLineColourAndStyle(-1,2);

    jetmassres_var_pu0.DrawAxes();
    jetmassres_var_pu0.AxisHisto()->GetYaxis()->SetRangeUser(0., 0.24);
    jetmassres_var_pu0.Draw( "same,HIST");
    jetmassres_var_pu02.Draw("same,HIST");
    jetmassres_var_pu08.Draw("same,HIST");


    placeLegend(legends::legend_pu_00_02_08, 0.6, 0.5)->Draw("same");
    //cv->SetLogy();
    cv->Print("jetmass_res_var.pdf");
    cv->SetLogy(false);


    jetmassres_var_pu0.renameAxes("m_{j}(t) [GeV]","mis-reco. fraction");
    jetmassres_var_pu0.DrawAxes();
    jetmassres_var_pu0.AxisHisto()->GetYaxis()->SetRangeUser(5e-4, 0.339);
    jetmassres_var_pu0.getOutlierFractionOC()->Draw( "same,HIST");
    jetmassres_var_pu02.getOutlierFractionOC()->Draw("same,HIST");
    jetmassres_var_pu08.getOutlierFractionOC()->Draw("same,HIST");
    jetmassres_var_pu0.getOutlierFractionPF()->Draw( "same,HIST");
    jetmassres_var_pu02.getOutlierFractionPF()->Draw("same,HIST");
    jetmassres_var_pu08.getOutlierFractionPF()->Draw("same,HIST");

    placeLegend(legends::legend_pu_00_02_08, 0.6, 0.55)->Draw("same");
    cv->SetLogy();
    cv->Print("jetmass_res_var_outliers.pdf");
    cv->SetLogy(false);

    for(const auto pu: pus){
        compareProfile jet_variance1_5("(jet_mass_r_"+pu+"/jet_mass_t_"+pu+" - 1)**2:jet_mass_t_"+pu+"",  " n_true <= 5",              30,0,1500,"m_{j}(t) [GeV]","Variance (m_{j})",true);
        compareProfile jet_variance5_10("(jet_mass_r_"+pu+"/jet_mass_t_"+pu+" - 1)**2:jet_mass_t_"+pu+"", " n_true <= 10 && n_true>5", 30,0,1500,"True momentum [GeV]","Variance",true);
        compareProfile jet_variance10_15("(jet_mass_r_"+pu+"/jet_mass_t_"+pu+" - 1)**2:jet_mass_t_"+pu+""," n_true ",                  30,0,1500,"True momentum [GeV]","Variance",true);


        jet_variance1_5.setClassicLineColourAndStyle(-1,3);
        jet_variance5_10.setClassicLineColourAndStyle(-1,2);
        jet_variance1_5.setOCLineColourAndStyle(-1,3);
        jet_variance5_10.setOCLineColourAndStyle(-1,2);


        jet_variance1_5.DrawAxes();
        //jet_variance1_5.AxisHisto()->GetYaxis()->SetRangeUser(0, 0.155);
        //jet_variance1_5.Draw("same");
        //jet_variance5_10.Draw("same");
        jet_variance10_15.Draw("same");

        placeLegend(legends::legend_onlyhi, 0.6, 0.5)->Draw("same");

        cv->Print("jetmass_variance"+pu+".pdf");

    }

    /*



     */





}
