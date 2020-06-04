#include "TH3D.h"

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

void setAxis(TAxis* ax, TString title){
    ax->SetTitle(title);
    ax->SetLabelColor(kWhite);
    ax->SetTitleColor(kWhite);
    ax->SetAxisColor(kWhite);
    ax->SetLabelSize(0.045);
   // ax->SetLabelOffset(.01);
    ax->SetTitleSize(0.05);
    ax->SetTitleOffset(1.5);
    ax->SetNdivisions(504);
}

int main(){

    TString inputfile="~/eos_miniCalo/PFTest4/929.root";


    TFile iTFile(inputfile,"READ");
    TTree * tree = (TTree*)iTFile.Get("B4");

    TCanvas cv("c","c",700,600);
    cv.GetPad(0)->SetFillColor(kBlack);
    cv.SetLeftMargin(0.15);
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);


    TH3D * h = new TH3D("h","h",10,-250,250,10,-250,250,10,-50,0);
    tree->SetMarkerSize(0.5);
    tree->SetMarkerStyle(10);
    tree->Draw("rechit_x:rechit_y:rechit_z:log(rechit_energy+1)>>h","","",1,0);

    setAxis(h->GetXaxis(),"z [mm]");
    setAxis(h->GetYaxis(),"x [mm]");
    setAxis(h->GetZaxis(),"y [mm]");


    h->Draw("");
    cv.SetLeftMargin(0.15);
    cv.Draw();

    cv.Print("geometry.pdf");

}
