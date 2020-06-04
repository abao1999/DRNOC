
#include "TFile.h"
#include "TString.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH2.h"
#include "TTree.h"
#include <iostream>
#include "TEfficiency.h"
#include "TStyle.h"
#include "TProfile.h"
#include "TLine.h"
#include "../interface/helpers.h"

#include "../interface/globals.h"

int main(int argc, char* argv[]){
    //if(argc<2) return -1;

    TString classic_files = "/afs/cern.ch/user/j/jkiesele/eos_miniCalo/PFTest4/test/forPFcalib/pf_cluster_calibration.root";

    TFile classic_f(classic_files,"READ");

    auto tree = (TTree*)classic_f.Get("tree");


    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    TCanvas *cv=createCanvas();

    TProfile * prof=new TProfile("prof","prof",200,0,200);
   // TProfile * prof=new TProfile("prof","prof",50,0,200);

    tree->Draw("corr_clust_energies/true_energies:clust_energies>>prof","","");
   // tree->Draw("true_energies/clust_energies:clust_energies>>prof","","");

    prof->GetYaxis()->SetTitle("E_{corr} / E_{true}");
    prof->GetXaxis()->SetTitle("E_{true} [GeV]");

    prof->GetXaxis()->SetLabelSize(0.05);
    prof->GetYaxis()->SetLabelSize(0.05);
    prof->GetXaxis()->SetTitleSize(0.05);
    prof->GetYaxis()->SetTitleSize(0.05);

    prof->GetYaxis()->SetRangeUser(0.9,1.1);

    prof->SetLineWidth(2);

    TLine* l = new TLine(0,1,200,1);
    l->SetLineStyle(2);


    prof->Draw();
    l->Draw("same");
    prof->Draw("same");


    cv->Print("pf_cluster_calibration.pdf");

    std::cout << "nbins " << prof->GetNbinsX() << "\n[" << std::endl;
    for(int i=1;i<prof->GetNbinsX()+1;i++){
        std::cout << prof->GetBinContent(i) << ", ";
    }
    std::cout << "]" << std::endl;




}
