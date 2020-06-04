

#include "TFile.h"
#include "TString.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH2.h"
#include "TTree.h"
#include <iostream>
#include "TEfficiency.h"
#include "TStyle.h"
#include "globals.h"
#include "TLegend.h"
#include "TLegendEntry.h"
#include "TProfile.h"

#include "TTreeReader.h"
#include "TTreeReaderValue.h"

namespace legends{
extern TLegend* legend_full;
extern TLegend* legend_smaller;
extern TLegend* legend_onlyex;
extern TLegend* legend_onlyhi;
extern TLegend* legend_onlyn;
extern TLegend* legend_simplest;

extern TLegend* legend_pu_00_02_08;
void buildLegends();
}

TCanvas * createCanvas();

class baseplotter{
public:
    baseplotter(TString var, TString selection, TString selectionpass, int nbins, double minbin, double maxbin,
            TString xaxis, TString yaxis){
        objstr_="";
        objstr_+=counter;
        counter++;
        axes_=0;
        createAxes(nbins,minbin,maxbin,xaxis,yaxis);
    }

    void DrawAxes(){
        axes_->Draw("AXIS");
    }

    void renameAxes(TString xaxis, TString yaxis){
        axes_->GetXaxis()->SetTitle(xaxis);
        axes_->GetYaxis()->SetTitle(yaxis);
    }

    TH1D* AxisHisto(){return axes_;}
protected:
    TH1D* axes_;
    TString objstr_;
    void createAxes(int nbins, double minbin, double maxbin,TString xaxis, TString yaxis){
        if (axes_)
            delete axes_;
        axes_ = new TH1D("axis"+objstr_,"axis"+objstr_,nbins,minbin,maxbin);

        axes_->GetXaxis()->SetTitle(xaxis);
        axes_->GetYaxis()->SetTitle(yaxis);
        axes_->GetYaxis()->SetTitleOffset(1.45);
        axes_->GetXaxis()->SetTitleOffset(1.15);

        axes_->GetXaxis()->SetLabelSize(0.05);
        axes_->GetYaxis()->SetLabelSize(0.05);
        axes_->GetXaxis()->SetTitleSize(0.05);
        axes_->GetYaxis()->SetTitleSize(0.05);


    }
    void mergeOverflows(TH1D* h)const{
        auto UF = h->GetBinContent(0);
        h->SetBinContent(0,0);
        h->SetBinContent(1,UF+h->GetBinContent(1));
        auto OF = h->GetBinContent(h->GetNbinsX()+1);
        h->SetBinContent(h->GetNbinsX()+1,0);
        h->SetBinContent(h->GetNbinsX(),OF+h->GetBinContent(h->GetNbinsX()));
    }
private:

    static int counter;
};



template<class T>
class comparePlotWithAxes: public baseplotter {
public:
    comparePlotWithAxes(TString var, TString selection, TString selectionpass,
            int nbins, double minbin, double maxbin, TString xaxis,
            TString yaxis, bool event=false) :
            baseplotter(var, selection, selectionpass, nbins, minbin, maxbin,
                    xaxis, yaxis) {
        event_=event;
        classic_o_ = 0;
        oc_o_ = 0;
    }
    virtual ~comparePlotWithAxes(){
        delete classic_o_;
        delete oc_o_;
    }

    void setClassicLineColourAndStyle(int col,int style=-1000){
        if(col>=0)
            (classic_o_)->SetLineColor(col);
        if(style>-1000)
            (classic_o_)->SetLineStyle(style);
    }
    void setOCLineColourAndStyle(int col,int style=-1000){
        if(col>=0)
            (oc_o_)->SetLineColor(col);
        if(style>-1000)
            (oc_o_)->SetLineStyle(style);
    }

    T* getOC(){return oc_o_;}
    T* getCl(){return classic_o_;}

    virtual void createObj(TString var, TString selection, TString selectionpass, int nbins, double minbin, double maxbin)=0;//{}

    void Draw(TString opt="", TString which=""){//always draws with same
        if(which.Contains("OC")){
            (oc_o_)->Draw(opt);
        }
        if(which.Contains("PF")){
            (classic_o_)->Draw(opt);
        }
        if(which.Length()<1){
            (classic_o_)->Draw(opt);
            (oc_o_)->Draw(opt);
        }

    }

    void setStyleDefaults(){
        (classic_o_)->SetLineColor(global::defaultClassicColour);
        (classic_o_)->SetLineWidth(2);
        (oc_o_)->SetLineColor(global::defaultOCColour);
        (oc_o_)->SetLineWidth(2);
    }



protected:

    T* classic_o_;
    T* oc_o_;

    std::vector<TObject*> otherobj_;
    bool event_;
};

class compareJetMassRes:public comparePlotWithAxes<TH1D> {
public:
    compareJetMassRes(int nbins, double minbin, double maxbin, TString xaxis, TString yaxis,TString pu,bool offset):
        comparePlotWithAxes("","","",nbins,minbin,maxbin,xaxis,yaxis),
        offset_(offset)
        {
        createObj("",pu,"",nbins,minbin,maxbin);

        setStyleDefaults();
    }

    void setStyleDefaults(){
        comparePlotWithAxes<TH1D>::setStyleDefaults();
        (outliers_pf_)->SetLineColor(global::defaultClassicColour);
        (outliers_pf_)->SetLineWidth(2);
        (outliers_oc_)->SetLineColor(global::defaultOCColour);
        (outliers_oc_)->SetLineWidth(2);
    }

    void setClassicLineColourAndStyle(int col,int style=-1000){
        comparePlotWithAxes<TH1D>::setClassicLineColourAndStyle(col,style);
        if(col>=0)
            (outliers_pf_)->SetLineColor(col);
        if(style>-1000)
            (outliers_pf_)->SetLineStyle(style);
    }
    void setOCLineColourAndStyle(int col,int style=-1000){
        comparePlotWithAxes<TH1D>::setOCLineColourAndStyle(col,style);
        if(col>=0)
            (outliers_oc_)->SetLineColor(col);
        if(style>-1000)
            (outliers_oc_)->SetLineStyle(style);
    }

    void createObj(TString var, TString pu, TString selectionpass, int nbins, double minbin, double maxbin) override {

        classic_o_ = loop(global::classic_event_tree, "cl"+objstr_,nbins,minbin,maxbin,pu,outliers_pf_);
        oc_o_ = loop(global::oc_event_tree, "oc"+objstr_,nbins,minbin,maxbin,pu,outliers_oc_);

       // mergeOverflows(classic_o_);
       // mergeOverflows(oc_o_);


        AxisHisto()->GetYaxis()->SetRangeUser(oc_o_->GetMaximum()*1e-9, oc_o_->GetMaximum()*1.1);
    }

    TH1D* getOutlierFractionOC(){return outliers_oc_;}
    TH1D* getOutlierFractionPF(){return outliers_pf_;}

private:
    TH1D * loop(TTree* t, TString histname, int nbins, double minbin, double maxbin,TString pu,TH1D*& outliers_) const{



        TH1D * h_av  = new TH1D(histname+"av",histname+"av", nbins,minbin,maxbin);
        h_av->Sumw2(true);
        TH1D * h_var = new TH1D(histname+"var",histname+"var", nbins,minbin,maxbin);
        h_var->Sumw2(true);
        TH1D * h_N = new TH1D(histname+"N",histname+"N", nbins,minbin,maxbin);
        h_N->Sumw2(true);
        outliers_ =  new TH1D(histname+"Nout",histname+"Nout", nbins,minbin,maxbin);
        outliers_->Sumw2(true);

        TTreeReader myReader(t);
        TTreeReaderValue<float> m_r(myReader, "jet_mass_r_"+pu);
        TTreeReaderValue<float> m_t(myReader, "jet_mass_t_"+pu);
        while (myReader.Next()) {
            if(! *m_t) continue;
            double ratio = *m_r / *m_t;
            if(fabs(ratio-1.)>0.8){
                outliers_->Fill(*m_t);
                continue;
            }
            h_av->Fill(*m_t,ratio);
            h_N->Fill(*m_t);
        }
        for(int i=0;i<h_N->GetNbinsX()+2;i++){
            if(h_N->GetBinContent(i) < 1){
                h_N->SetBinContent(i,1.);
                h_N->SetBinError(i,1.);
            }
            double norm = h_N->GetBinContent(i);
            h_av->SetBinContent(i, h_av->GetBinContent(i) / norm);
            h_av->SetBinError(i, h_av->GetBinError(i) / norm);
            if(h_av->GetBinContent(i)!=h_av->GetBinContent(i)){
                h_av->SetBinContent(i,1.);
                h_av->SetBinError(i,1.);
            }
            outliers_->SetBinContent(i, outliers_->GetBinContent(i) / norm);
            if(outliers_->GetBinContent(i)!=outliers_->GetBinContent(i)){
                outliers_->SetBinContent(i,0.);
                outliers_->SetBinError(i,0.);
            }
        }
       // h_av->Divide(h_N);//h_av,h_N,1.,1.,"B");

        myReader.Restart();
        while (myReader.Next()) {
            int bin=h_av->FindBin(*m_t);
            if(! *m_t) continue;
            double ratio = *m_r / *m_t;
            if(fabs(ratio-1.)>0.8){
                continue;
            }
            double var = (ratio - h_av->GetBinContent(bin));
            var*=var;
            h_var->Fill(*m_t, var);

        }

        h_var->Divide(h_N);//,1.,1.,"B");
        for(int i=0; i<h_av->GetNbinsX()+1; i++){
            if (h_var->GetBinContent(i) == h_var->GetBinContent(i)){//not nan
                h_var->SetBinContent(i, sqrt(h_var->GetBinContent(i)));
                h_var->SetBinError(i, sqrt(h_var->GetBinError(i)));
            }
            else
                h_var->SetBinContent(i, 0);
        }
        if(offset_)
            return h_av;
        else
            return h_var;
    }
    bool offset_;
    TH1D* outliers_oc_,*outliers_pf_;
};


class compareEfficiency: public comparePlotWithAxes<TEfficiency> {
public:
    compareEfficiency(TString var, TString selection, TString selectionpass, int nbins, double minbin, double maxbin, TString xaxis, TString yaxis):
        comparePlotWithAxes(var,selection,selectionpass,nbins,minbin,maxbin,xaxis,yaxis){
        createObj(var,selection,selectionpass,nbins,minbin,maxbin);
        setStyleDefaults();
    }

    void createObj(TString var, TString selection, TString selectionpass, int nbins, double minbin, double maxbin) override {

        classic_o_ = makeEfficiency(global::classic_tree, "cl"+objstr_,var, selection,selectionpass,nbins,minbin,maxbin);
        oc_o_ = makeEfficiency(global::oc_tree, "oc"+objstr_,var, selection,selectionpass,nbins,minbin,maxbin);

    }

private:
    TEfficiency * makeEfficiency(TTree* t, TString add, TString var, TString selection, TString selectionpass, int nbins, double minbin, double maxbin){

        TH1D * hpass  = new TH1D("hpass"+add,"hpass"+add, nbins, minbin, maxbin);
        otherobj_.push_back(hpass);
        TH1D * htotal = new TH1D("htotal"+add,"htotal"+add, nbins, minbin, maxbin);
        otherobj_.push_back(htotal);

        TEfficiency* eff = new TEfficiency("eff"+add,"eff"+add,nbins,minbin,maxbin);
        eff->SetUseWeightedEvents(true);
        eff->SetStatisticOption(TEfficiency::kFNormal);

        std::cout << var+">>"+"htotal"+add << std::endl;

        t->Draw(var+">>"+"htotal"+add,selection);
        t->Draw(var+">>"+"hpass"+add,selectionpass);

        eff->SetTotalHistogram(*htotal,"");
        eff->SetPassedHistogram(*hpass,"");

        return eff;
    }

};

class compareTH1D: public comparePlotWithAxes<TH1D> {
public:
    compareTH1D(TString var, TString selection, int nbins, double minbin, double maxbin, TString xaxis, TString yaxis,bool normalize=true):
        comparePlotWithAxes(var,selection,"",nbins,minbin,maxbin,xaxis,yaxis),
        normalize_(normalize){
        createObj(var,selection,"",nbins,minbin,maxbin);
        setStyleDefaults();
    }

    void createObj(TString var, TString selection, TString selectionpass, int nbins, double minbin, double maxbin) override {

        classic_o_  = new TH1D("ha"+objstr_,"ha"+objstr_, nbins, minbin, maxbin);
        oc_o_  = new TH1D("hb"+objstr_,"hb"+objstr_, nbins, minbin, maxbin);

        global::classic_tree->Draw(var+">>"+"ha"+objstr_,selection);
        global::oc_tree->Draw(var+">>"+"hb"+objstr_,selection);

        mergeOverflows(classic_o_);
        mergeOverflows(oc_o_);

        if(normalize_){
            classic_o_->Scale(1./classic_o_->Integral());
            oc_o_->Scale(1./oc_o_->Integral());
        }

        auto max=oc_o_->GetMaximum();

        AxisHisto()->GetYaxis()->SetRangeUser(0, max*1.1);

    }
private:
    bool normalize_;
};



class compareProfile: public comparePlotWithAxes<TProfile> {
public:
    compareProfile(TString var, TString selection, int nbins, double minbin, double maxbin, TString xaxis, TString yaxis,bool event=false):
        comparePlotWithAxes(var,selection,"",nbins,minbin,maxbin,xaxis,yaxis,event){
        createObj(var,selection,"",nbins,minbin,maxbin);
        setStyleDefaults();
    }

    void createObj(TString var, TString selection, TString selectionpass, int nbins, double minbin, double maxbin) override {

        classic_o_  = new TProfile("ha"+objstr_,"ha"+objstr_, nbins, minbin, maxbin);
        oc_o_       = new TProfile("hb"+objstr_,"hb"+objstr_, nbins, minbin, maxbin);


        if(event_){
            global::classic_event_tree->Draw(var+">>"+"ha"+objstr_,selection,"prof");
            global::oc_event_tree->Draw(var+">>"+"hb"+objstr_,selection,"prof");
        }
        else{
            global::classic_tree->Draw(var+">>"+"ha"+objstr_,selection,"prof");
            global::oc_tree->Draw(var+">>"+"hb"+objstr_,selection,"prof");
        }
        auto max=oc_o_->GetMaximum();

        AxisHisto()->GetYaxis()->SetRangeUser(0, max*1.1);

    }
private:

};



TLegendEntry * makeLegEntry(TLegend* leg, TString name, TString option, int col, int style=-1);


TLegend* placeLegend(TLegend* leg, double x1, double y1, double x2=-1, double y2=-1);






